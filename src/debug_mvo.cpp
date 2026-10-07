// Detailed MVO debug test
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <nlopt.h>
#include <cmath>

// Split string
std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Load data
Eigen::MatrixXd loadData(const std::string& csv_path, std::vector<std::string>& dates) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << csv_path << std::endl;
        return Eigen::MatrixXd(0, 0);
    }
    
    std::string line;
    std::vector<std::vector<double>> prices;
    
    std::getline(file, line);  // Skip header
    
    while (std::getline(file, line)) {
        auto tokens = split(line, ',');
        if (tokens.size() < 2) continue;
        
        dates.push_back(tokens[0]);
        
        std::vector<double> price_row;
        for (size_t i = 1; i < tokens.size(); ++i) {
            price_row.push_back(std::stod(tokens[i]));
        }
        prices.push_back(price_row);
    }
    
    file.close();
    
    int n_days = prices.size();
    int n_assets = prices[0].size();
    
    Eigen::MatrixXd returns(n_days, n_assets);
    for (int i = 0; i < n_days; ++i) {
        for (int j = 0; j < n_assets; ++j) {
            if (i == 0) {
                returns(i, j) = 0.0;
            } else {
                double prev_price = prices[i-1][j];
                double curr_price = prices[i][j];
                if (std::abs(prev_price) > 1e-10) {
                    returns(i, j) = curr_price / prev_price - 1.0;
                } else {
                    returns(i, j) = 0.0;
                }
            }
        }
    }
    
    return returns;
}

// 3-sigma preprocessing
Eigen::MatrixXd apply3Sigma(const Eigen::MatrixXd& returns, int window, int n_sigma) {
    int n_days = returns.rows();
    int n_assets = returns.cols();
    Eigen::MatrixXd result = returns;
    
    for (int i = window; i < n_days; ++i) {
        Eigen::VectorXd mean(n_assets);
        Eigen::VectorXd std_vec(n_assets);
        
        for (int j = 0; j < n_assets; ++j) {
            double sum = 0.0, sum_sq = 0.0;
            for (int k = i - window; k < i; ++k) {
                double val = result(k, j);
                sum += val;
                sum_sq += val * val;
            }
            double n = static_cast<double>(window);
            mean(j) = sum / n;
            double variance = (sum_sq - sum * sum / n) / n;
            std_vec(j) = (variance > 0) ? std::sqrt(variance) : 1e-10;
        }
        
        for (int j = 0; j < n_assets; ++j) {
            double upper = mean(j) + n_sigma * std_vec(j);
            double lower = mean(j) - n_sigma * std_vec(j);
            result(i, j) = std::clamp(result(i, j), lower, upper);
        }
    }
    
    return result;
}

// Rolling sum
Eigen::MatrixXd rollingSum(const Eigen::MatrixXd& returns, int keep) {
    int n_days = returns.rows();
    int n_assets = returns.cols();
    Eigen::MatrixXd result(n_days - keep + 1, n_assets);
    
    for (int i = 0; i <= n_days - keep; ++i) {
        result.row(i).setZero();
        for (int k = 0; k < keep; ++k) {
            result.row(i) += returns.row(i + k);
        }
    }
    
    return result;
}

// Equal weight covariance
Eigen::MatrixXd equalWeightCov(const Eigen::MatrixXd& returns) {
    int n = returns.rows();
    int d = returns.cols();
    
    Eigen::VectorXd mean = returns.colwise().mean();
    Eigen::MatrixXd centered = returns.rowwise() - mean.transpose();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / n;
    
    return cov;
}

// Equal weight std
Eigen::VectorXd equalWeightStd(const Eigen::MatrixXd& returns) {
    return returns.colwise().stableNorm() / std::sqrt(returns.rows());
}

// Max diversification objective: -w'σ / sqrt(w'Σw)
double evalMaxDiv(const Eigen::VectorXd& w, const Eigen::VectorXd& std_vec, const Eigen::MatrixXd& cov) {
    double w_std = w.dot(std_vec);
    double w_cov_w = w.dot(cov * w);
    double denom = std::sqrt(w_cov_w);
    if (denom < 1e-10) return 0.0;
    return -w_std / denom;
}

// Gradient of max diversification
void gradMaxDiv(const Eigen::VectorXd& w, const Eigen::VectorXd& std_vec, const Eigen::MatrixXd& cov, Eigen::VectorXd& grad) {
    double w_std = w.dot(std_vec);
    double w_cov_w = w.dot(cov * w);
    double denom = std::sqrt(w_cov_w);
    double denom_cubed = denom * denom * denom;
    
    if (denom < 1e-10) {
        grad.setZero();
        return;
    }
    
    grad = -std_vec / denom + (w_std / denom_cubed) * (cov * w);
}

// Constraint: sum(w) = 1
double eqSumOne(const Eigen::VectorXd& w) {
    return w.sum() - 1.0;
}

void eqSumOneGrad(const Eigen::VectorXd& w, Eigen::VectorXd& grad) {
    grad = Eigen::VectorXd::Ones(w.size());
}

// Constraint: turnover limit
double ineqTurnover(const Eigen::VectorXd& w, const Eigen::VectorXd& prev_w, double limit) {
    return limit - (w - prev_w).cwiseAbs().sum();
}

void ineqTurnoverGrad(const Eigen::VectorXd& w, const Eigen::VectorXd& prev_w, double limit, Eigen::VectorXd& grad) {
    grad = -(w - prev_w).cwiseSign();
}

// Group constraint
double ineqGroupLower(const Eigen::VectorXd& w, int start, int end, double min_val) {
    return w.segment(start, end - start).sum() - min_val;
}

double ineqGroupUpper(const Eigen::VectorXd& w, int start, int end, double max_val) {
    return max_val - w.segment(start, end - start).sum();
}

void ineqGroupGrad(const Eigen::VectorXd& w, int start, int end, Eigen::VectorXd& grad) {
    grad.setZero(w.size());
    for (int i = start; i < end; ++i) {
        grad(i) = 1.0;
    }
}

// Single asset constraint
double ineqAssetLower(const Eigen::VectorXd& w, int idx, double min_val) {
    return w(idx) - min_val;
}

double ineqAssetUpper(const Eigen::VectorXd& w, int idx, double max_val) {
    return max_val - w(idx);
}

int main() {
    std::string data_path = "/home/chow/quant/suishi-quant/data/data.csv";
    std::vector<std::string> dates;
    
    std::cout << "Loading data...\n";
    Eigen::MatrixXd returns = loadData(data_path, dates);
    std::cout << "Data: " << returns.rows() << " days, " << returns.cols() << " assets\n";
    
    int window = 60;
    int keep = 15;
    int n_sigma = 3;
    
    std::cout << "\nApplying 3-sigma preprocessing...\n";
    Eigen::MatrixXd processed = apply3Sigma(returns, window, n_sigma);
    
    // Initial weights
    Eigen::VectorXd init_weight(8);
    init_weight << 0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306;
    std::cout << "Initial weights:\n" << init_weight.transpose() << "\n\n";
    
    // Test optimization at day 75 (first optimization point)
    int opt_day = window;  // First optimization day
    std::cout << "First optimization at day " << opt_day << " (" << dates[opt_day] << ")\n";
    
    // Get window returns
    Eigen::MatrixXd window_returns = processed.block(opt_day - window, 0, window, 8);
    window_returns = rollingSum(window_returns, keep);
    
    std::cout << "Window returns shape: " << window_returns.rows() << " x " << window_returns.cols() << "\n";
    
    // Calculate covariance
    Eigen::MatrixXd cov = equalWeightCov(window_returns);
    Eigen::VectorXd std_vec = equalWeightStd(window_returns);
    Eigen::VectorXd mean_ret = window_returns.colwise().mean();
    
    std::cout << "\nStd vector:\n" << std_vec.transpose() << "\n";
    std::cout << "\nMean returns:\n" << mean_ret.transpose() << "\n";
    
    // Test objective at initial weights
    double obj_init = evalMaxDiv(init_weight, std_vec, cov);
    std::cout << "\nObjective at initial weights: " << obj_init << "\n";
    
    // Setup NLopt
    int n = 8;
    nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, n);
    
    // Objective data
    struct ObjData {
        const Eigen::VectorXd* std_vec;
        const Eigen::MatrixXd* cov;
    };
    ObjData obj_data{&std_vec, &cov};
    
    auto obj_func = [](unsigned n, const double* x, double* grad, void* data) -> double {
        auto* d = static_cast<ObjData*>(data);
        Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
        
        if (grad) {
            Eigen::VectorXd g(n);
            gradMaxDiv(w, *d->std_vec, *d->cov, g);
            for (unsigned i = 0; i < n; ++i) {
                grad[i] = g(i);
            }
        }
        return evalMaxDiv(w, *d->std_vec, *d->cov);
    };
    
    nlopt_set_min_objective(opt, obj_func, &obj_data);
    
    // Add constraints
    auto ineq_func_c = [](unsigned n, const double* x, double* grad, void* data) -> double {
        auto* d = static_cast<std::pair<double, int>*>(data);
        double val = d->first;
        if (grad) {
            grad[d->second] = -1.0;
        }
        return -val;
    };
    
    auto ineq_turnover_func = [](unsigned n, const double* x, double* grad, void* data) -> double {
        auto* d = static_cast<std::pair<double, const Eigen::VectorXd*>*>(data);
        double limit = d->first;
        const Eigen::VectorXd& prev = *d->second;
        Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
        
        if (grad) {
            for (unsigned i = 0; i < n; ++i) {
                grad[i] = -(w(i) - prev(i) > 0 ? 1.0 : -1.0);
            }
        }
        return limit - (w - prev).cwiseAbs().sum();
    };
    
    // Equality: sum(w) = 1
    nlopt_add_equality_constraint(opt, 
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (unsigned i = 0; i < n; ++i) grad[i] = 1.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return w.sum() - 1.0;
        }, nullptr, 1e-8);
    
    // Turnover constraint
    auto turnover_data = new std::pair<double, const Eigen::VectorXd*>(0.1, &init_weight);
    nlopt_add_inequality_constraint(opt, ineq_turnover_func, turnover_data, 1e-8);
    
    // Group 1 constraints (w[0:4] in [0.15, 0.35])
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 4; ++i) grad[i] = 1.0;
                for (int i = 4; i < 8; ++i) grad[i] = 0.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return w.segment(0, 4).sum() - 0.15;
        }, nullptr, 1e-8);
    
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 4; ++i) grad[i] = -1.0;
                for (int i = 4; i < 8; ++i) grad[i] = 0.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return 0.35 - w.segment(0, 4).sum();
        }, nullptr, 1e-8);
    
    // Asset 2 constraints (w[4] in [0.15, 0.35])
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 8; ++i) grad[i] = (i == 4) ? 1.0 : 0.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return w(4) - 0.15;
        }, nullptr, 1e-8);
    
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 8; ++i) grad[i] = (i == 4) ? -1.0 : 0.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return 0.35 - w(4);
        }, nullptr, 1e-8);
    
    // Asset 3 constraints (w[5] in [0.15, 0.35])
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 8; ++i) grad[i] = (i == 5) ? 1.0 : 0.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return w(5) - 0.15;
        }, nullptr, 1e-8);
    
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 8; ++i) grad[i] = (i == 5) ? -1.0 : 0.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return 0.35 - w(5);
        }, nullptr, 1e-8);
    
    // Group 2 constraints (w[6:8] in [0.15, 0.35])
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 6; ++i) grad[i] = 0.0;
                for (int i = 6; i < 8; ++i) grad[i] = 1.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return w.segment(6, 2).sum() - 0.15;
        }, nullptr, 1e-8);
    
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 6; ++i) grad[i] = 0.0;
                for (int i = 6; i < 8; ++i) grad[i] = -1.0;
            }
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            return 0.35 - w.segment(6, 2).sum();
        }, nullptr, 1e-8);
    
    // Bounds
    std::vector<double> lb(n, 0.0);
    nlopt_set_lower_bounds(opt, lb.data());
    
    // Settings
    nlopt_set_maxeval(opt, 1000);
    nlopt_set_ftol_abs(opt, 1e-10);
    nlopt_set_ftol_rel(opt, 1e-10);
    
    // Optimize
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = init_weight(i);
    
    double minf;
    int result = nlopt_optimize(opt, x.data(), &minf);
    
    std::cout << "\nOptimization result code: " << result << "\n";
    std::cout << "Optimal weights:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "  w[" << i << "] = " << x[i] << "\n";
    }
    
    Eigen::VectorXd opt_weight = Eigen::Map<Eigen::VectorXd>(x.data(), n);
    double obj_opt = evalMaxDiv(opt_weight, std_vec, cov);
    std::cout << "\nObjective at optimal: " << obj_opt << "\n";
    std::cout << "Sum of weights: " << opt_weight.sum() << "\n";
    std::cout << "Turnover: " << (opt_weight - init_weight).cwiseAbs().sum() << "\n";
    
    nlopt_destroy(opt);
    delete turnover_data;
    
    return 0;
}
