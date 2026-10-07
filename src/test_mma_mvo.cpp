// Test NLopt MMA with actual MVO objective
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <nlopt.h>
#include <cmath>

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delim)) tokens.push_back(token);
    return tokens;
}

Eigen::MatrixXd loadData(const std::string& csv_path, std::vector<std::string>& dates) {
    std::ifstream file(csv_path);
    if (!file.is_open()) return Eigen::MatrixXd(0, 0);
    
    std::string line;
    std::vector<std::vector<double>> prices;
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        auto tokens = split(line, ',');
        if (tokens.size() < 2) continue;
        dates.push_back(tokens[0]);
        std::vector<double> row;
        for (size_t i = 1; i < tokens.size(); ++i) row.push_back(std::stod(tokens[i]));
        prices.push_back(row);
    }
    
    int n_days = prices.size(), n_assets = prices[0].size();
    Eigen::MatrixXd returns(n_days, n_assets);
    for (int i = 0; i < n_days; ++i)
        for (int j = 0; j < n_assets; ++j)
            returns(i, j) = (i == 0) ? 0.0 : (prices[i][j] / prices[i-1][j] - 1.0);
    return returns;
}

// 3-sigma
Eigen::MatrixXd apply3Sigma(const Eigen::MatrixXd& r, int window, int n_sigma) {
    Eigen::MatrixXd result = r;
    for (int i = window; i < r.rows(); ++i) {
        for (int j = 0; j < r.cols(); ++j) {
            double sum = 0, sum_sq = 0;
            for (int k = i - window; k < i; ++k) { sum += result(k, j); sum_sq += result(k, j) * result(k, j); }
            double n = window, mean = sum / n, variance = (sum_sq - sum * sum / n) / n, std = sqrt(variance);
            result(i, j) = std::clamp(result(i, j), mean - n_sigma * std, mean + n_sigma * std);
        }
    }
    return result;
}

int main() {
    std::vector<std::string> dates;
    Eigen::MatrixXd returns = loadData("/home/chow/quant/suishi-quant/data/data.csv", dates);
    Eigen::MatrixXd processed = apply3Sigma(returns, 60, 3);
    
    // Get window returns at day 60
    Eigen::MatrixXd window = processed.block(0, 0, 60, 8);
    Eigen::MatrixXd window_sum(46, 8);
    for (int i = 0; i <= 46 - 15; ++i) {
        window_sum.row(i).setZero();
        for (int k = 0; k < 15; ++k) window_sum.row(i) += window.row(i + k);
    }
    
    // Calculate cov
    Eigen::VectorXd mean = window_sum.colwise().mean();
    Eigen::MatrixXd centered = window_sum.rowwise() - mean.transpose();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / 46;
    Eigen::VectorXd std_vec = cov.diagonal().cwiseSqrt();
    
    double prev_w[8] = {0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306};
    int n = 8;
    
    struct MyData { Eigen::VectorXd mean; Eigen::VectorXd std_vec; Eigen::MatrixXd cov; };
    MyData* data = new MyData{mean, std_vec, cov};
    
    // Objective: max diversification -w'σ / sqrt(w'Σw)
    auto obj_func = [](unsigned n, const double* x, double* grad, void* data) -> double {
        auto* d = static_cast<MyData*>(data);
        Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
        if (grad) {
            double w_std = w.dot(d->std_vec);
            double w_cov_w = w.dot(d->cov * w);
            double denom = sqrt(w_cov_w), denom_cubed = denom * denom * denom;
            Eigen::VectorXd g = -d->std_vec / denom + (w_std / denom_cubed) * (d->cov * w);
            for (unsigned i = 0; i < n; ++i) grad[i] = g(i);
        }
        double w_std = w.dot(d->std_vec);
        double w_cov_w = w.dot(d->cov * w);
        return -w_std / sqrt(w_cov_w);
    };
    
    nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, n);
    nlopt_set_min_objective(opt, obj_func, data);
    
    // Equality: sum(w) = 1
    nlopt_add_equality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < n; ++i) grad[i] = 1.0;
        double s = 0; for (int i = 0; i < n; ++i) s += x[i]; return s - 1.0;
    }, nullptr, 1e-10);
    
    // Turnover
    auto turnover_data = new std::pair<double, double*>(0.1, prev_w);
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void* data) -> double {
        auto* d = static_cast<std::pair<double, double*>*>(data);
        if (grad) for (int i = 0; i < n; ++i) grad[i] = -(x[i] > d->second[i] ? 1.0 : -1.0);
        double s = 0; for (int i = 0; i < n; ++i) s += std::abs(x[i] - d->second[i]); return 0.1 - s;
    }, turnover_data, 1e-8);
    
    // Group 1
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < 4; ++i) grad[i] = 1.0;
        return x[0]+x[1]+x[2]+x[3] - 0.15;
    }, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < 4; ++i) grad[i] = -1.0;
        return 0.35 - (x[0]+x[1]+x[2]+x[3]);
    }, nullptr, 1e-8);
    
    // Asset 2
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < n; ++i) grad[i] = (i == 4) ? 1.0 : 0.0;
        return x[4] - 0.15;
    }, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < n; ++i) grad[i] = (i == 4) ? -1.0 : 0.0;
        return 0.35 - x[4];
    }, nullptr, 1e-8);
    
    // Asset 3
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < n; ++i) grad[i] = (i == 5) ? 1.0 : 0.0;
        return x[5] - 0.15;
    }, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < n; ++i) grad[i] = (i == 5) ? -1.0 : 0.0;
        return 0.35 - x[5];
    }, nullptr, 1e-8);
    
    // Group 2
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < 6; ++i) grad[i] = 0.0;
        return x[6]+x[7] - 0.15;
    }, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt, [](unsigned n, const double* x, double* grad, void*) -> double {
        if (grad) for (int i = 0; i < 6; ++i) grad[i] = 0.0;
        return 0.35 - (x[6]+x[7]);
    }, nullptr, 1e-8);
    
    std::vector<double> lb(n, 0.0);
    nlopt_set_lower_bounds(opt, lb.data());
    nlopt_set_maxeval(opt, 1000);
    
    std::vector<double> x(8);
    for (int i = 0; i < n; ++i) x[i] = prev_w[i];
    
    double minf;
    int res = nlopt_optimize(opt, x.data(), &minf);
    
    std::cout << "MMA Result: " << res << "\n";
    std::cout << "Objective: " << minf << "\n";
    std::cout << "Weights:\n";
    double sum = 0, turnover = 0;
    for (int i = 0; i < n; ++i) {
        std::cout << "  w[" << i << "] = " << x[i] << "\n";
        sum += x[i];
        turnover += std::abs(x[i] - prev_w[i]);
    }
    std::cout << "\nSum: " << sum << "\n";
    std::cout << "Turnover: " << turnover << "\n";
    
    nlopt_destroy(opt);
    delete data;
    delete turnover_data;
    
    return 0;
}
