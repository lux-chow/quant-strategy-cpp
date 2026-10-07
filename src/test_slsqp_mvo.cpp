// Test with SLSQP
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <nlopt.h>
#include <cmath>

// Global functions for clarity
double evalMaxDiv(const Eigen::VectorXd& w, const Eigen::VectorXd& std_vec, const Eigen::MatrixXd& cov) {
    double w_std = w.dot(std_vec);
    double w_cov_w = w.dot(cov * w);
    double denom = std::sqrt(w_cov_w);
    return (denom < 1e-10) ? 0.0 : -w_std / denom;
}

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

int main() {
    std::cout << "Testing SLSQP with MVO objective...\n\n";
    
    int n = 8;
    double prev_w[8] = {0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306};
    
    // Simple test data
    Eigen::MatrixXd window_returns(46, 8);
    window_returns.setRandom();
    window_returns *= 0.01;
    
    // Calculate cov
    Eigen::VectorXd mean = window_returns.colwise().mean();
    Eigen::MatrixXd centered = window_returns.rowwise() - mean.transpose();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / 46;
    Eigen::VectorXd std_vec = cov.diagonal().cwiseSqrt();
    
    std::cout << "Test data std:\n" << std_vec.transpose() << "\n\n";
    
    // Objective data
    struct ObjData {
        Eigen::VectorXd std_vec;
        Eigen::MatrixXd cov;
    };
    ObjData* obj_data = new ObjData{std_vec, cov};
    
    std::cout << "=== Testing SLSQP ===\n";
    nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, n);
    
    // Set objective
    nlopt_set_min_objective(opt, 
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            auto* d = static_cast<ObjData*>(data);
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            
            if (grad) {
                Eigen::VectorXd g(n);
                gradMaxDiv(w, d->std_vec, d->cov, g);
                for (unsigned i = 0; i < n; ++i) grad[i] = g(i);
            }
            return evalMaxDiv(w, d->std_vec, d->cov);
        }, obj_data);
    
    // Equality: sum(w) = 1
    nlopt_add_equality_constraint(opt, 
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < n; ++i) grad[i] = 1.0;
            }
            double sum = 0;
            for (int i = 0; i < n; ++i) sum += x[i];
            return sum - 1.0;
        }, nullptr, 1e-10);
    
    // Turnover constraint
    auto turnover_data = new std::pair<double, double*>(0.1, prev_w);
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            auto* d = static_cast<std::pair<double, double*>*>(data);
            double limit = d->first;
            double* prev = d->second;
            if (grad) {
                for (int i = 0; i < n; ++i) {
                    grad[i] = -(x[i] > prev[i] ? 1.0 : -1.0);
                }
            }
            double sum = 0;
            for (int i = 0; i < n; ++i) sum += std::abs(x[i] - prev[i]);
            return limit - sum;
        }, turnover_data, 1e-8);
    
    // Group 1: w[0:4] in [0.15, 0.35]
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 4; ++i) grad[i] = 1.0;
                for (int i = 4; i < 8; ++i) grad[i] = 0.0;
            }
            return x[0] + x[1] + x[2] + x[3] - 0.15;
        }, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 4; ++i) grad[i] = -1.0;
                for (int i = 4; i < 8; ++i) grad[i] = 0.0;
            }
            return 0.35 - (x[0] + x[1] + x[2] + x[3]);
        }, nullptr, 1e-8);
    
    // Asset 2: w[4] in [0.15, 0.35]
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < n; ++i) grad[i] = (i == 4) ? 1.0 : 0.0;
            }
            return x[4] - 0.15;
        }, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < n; ++i) grad[i] = (i == 4) ? -1.0 : 0.0;
            }
            return 0.35 - x[4];
        }, nullptr, 1e-8);
    
    // Asset 3: w[5] in [0.15, 0.35]
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < n; ++i) grad[i] = (i == 5) ? 1.0 : 0.0;
            }
            return x[5] - 0.15;
        }, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < n; ++i) grad[i] = (i == 5) ? -1.0 : 0.0;
            }
            return 0.35 - x[5];
        }, nullptr, 1e-8);
    
    // Group 2: w[6:8] in [0.15, 0.35]
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 6; ++i) grad[i] = 0.0;
                for (int i = 6; i < 8; ++i) grad[i] = 1.0;
            }
            return x[6] + x[7] - 0.15;
        }, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt,
        [](unsigned n, const double* x, double* grad, void* data) -> double {
            if (grad) {
                for (int i = 0; i < 6; ++i) grad[i] = 0.0;
                for (int i = 6; i < 8; ++i) grad[i] = -1.0;
            }
            return 0.35 - (x[6] + x[7]);
        }, nullptr, 1e-8);
    
    // Bounds
    std::vector<double> lb(n, 0.0);
    nlopt_set_lower_bounds(opt, lb.data());
    nlopt_set_maxeval(opt, 2000);
    nlopt_set_ftol_abs(opt, 1e-12);
    nlopt_set_ftol_rel(opt, 1e-12);
    
    // Initial guess
    std::vector<double> x = {0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306};
    double minf;
    int result = nlopt_optimize(opt, x.data(), &minf);
    
    std::cout << "SLSQP Result: " << result << "\n";
    double sum = 0, turnover = 0;
    for (int i = 0; i < n; ++i) {
        std::cout << "w[" << i << "] = " << x[i] << "\n";
        sum += x[i];
        turnover += std::abs(x[i] - prev_w[i]);
    }
    std::cout << "Sum: " << sum << "\n";
    std::cout << "Turnover: " << turnover << "\n";
    
    nlopt_destroy(opt);
    delete obj_data;
    delete turnover_data;
    
    return 0;
}
