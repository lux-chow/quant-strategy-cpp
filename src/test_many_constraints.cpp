// Test NLopt with many constraints
#include <iostream>
#include <vector>
#include <nlopt.h>
#include <cmath>

int main() {
    std::cout << "Testing NLopt with 8 variables and 11 constraints...\n\n";
    
    int n = 8;
    
    // Simple quadratic objective
    auto obj_func = [](unsigned n, const double* x, double* grad, void* data) -> double {
        double val = 0;
        for (int i = 0; i < n; ++i) {
            double d = x[i] - 0.1;
            val += d * d;
            if (grad) grad[i] = 2 * d;
        }
        return val;
    };
    
    // Test MMA with all constraints
    std::cout << "=== Testing MMA with all constraints ===\n";
    {
        nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, n);
        nlopt_set_min_objective(opt, obj_func, nullptr);
        
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
        double prev_w[8] = {0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306};
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
        
        std::cout << "Result: " << result << "\n";
        double sum = 0, turnover = 0;
        for (int i = 0; i < n; ++i) {
            std::cout << "w[" << i << "] = " << x[i] << "\n";
            sum += x[i];
            turnover += std::abs(x[i] - prev_w[i]);
        }
        std::cout << "Sum: " << sum << "\n";
        std::cout << "Turnover: " << turnover << "\n";
        std::cout << "Group1: " << (x[0]+x[1]+x[2]+x[3]) << "\n";
        std::cout << "Asset2: " << x[4] << "\n";
        std::cout << "Asset3: " << x[5] << "\n";
        std::cout << "Group2: " << (x[6]+x[7]) << "\n";
        
        nlopt_destroy(opt);
        delete turnover_data;
    }
    
    std::cout << "\n\n=== Testing SLSQP with all constraints ===\n";
    {
        nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, n);
        nlopt_set_min_objective(opt, obj_func, nullptr);
        
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
        double prev_w[8] = {0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306};
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
        
        std::cout << "Result: " << result << "\n";
        double sum = 0, turnover = 0;
        for (int i = 0; i < n; ++i) {
            std::cout << "w[" << i << "] = " << x[i] << "\n";
            sum += x[i];
            turnover += std::abs(x[i] - prev_w[i]);
        }
        std::cout << "Sum: " << sum << "\n";
        std::cout << "Turnover: " << turnover << "\n";
        std::cout << "Group1: " << (x[0]+x[1]+x[2]+x[3]) << "\n";
        std::cout << "Asset2: " << x[4] << "\n";
        std::cout << "Asset3: " << x[5] << "\n";
        std::cout << "Group2: " << (x[6]+x[7]) << "\n";
        
        nlopt_destroy(opt);
        delete turnover_data;
    }
    
    return 0;
}
