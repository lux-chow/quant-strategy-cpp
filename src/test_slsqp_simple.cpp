// Test NLopt SLSQP constraint implementation
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <nlopt.h>
#include <cmath>

// Test objective
double test_obj(unsigned n, const double* x, double* grad, void* data) {
    if (grad) {
        for (int i = 0; i < n; ++i) {
            grad[i] = 2 * (x[i] - 0.1);
        }
    }
    double val = 0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - 0.1;
        val += d * d;
    }
    return val;
}

// Constraint: sum(w) = 1
double eq_sum(unsigned n, const double* x, double* grad, void* data) {
    if (grad) {
        for (int i = 0; i < n; ++i) grad[i] = 1.0;
    }
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += x[i];
    return sum - 1.0;
}

// Constraint: turnover limit
double ineq_turnover(unsigned n, const double* x, double* grad, void* data) {
    auto* d = static_cast<std::pair<double, std::vector<double>>*>(data);
    double limit = d->first;
    const std::vector<double>& prev = d->second;
    
    if (grad) {
        for (int i = 0; i < n; ++i) {
            grad[i] = -(x[i] > prev[i] ? 1.0 : -1.0);
        }
    }
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += std::abs(x[i] - prev[i]);
    return limit - sum;
}

int main() {
    int n = 8;
    std::vector<double> prev_w = {0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306};
    
    std::cout << "=== Testing SLSQP with constraints ===\n";
    
    nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, n);
    if (!opt) {
        std::cerr << "Failed to create optimizer!\n";
        return 1;
    }
    
    nlopt_set_min_objective(opt, test_obj, nullptr);
    
    // Equality constraint
    int res = nlopt_add_equality_constraint(opt, eq_sum, nullptr, 1e-10);
    std::cout << "add_equality_constraint result: " << res << "\n";
    
    // Turnover constraint
    auto* turnover_data = new std::pair<double, std::vector<double>>(0.1, prev_w);
    res = nlopt_add_inequality_constraint(opt, ineq_turnover, turnover_data, 1e-8);
    std::cout << "add_inequality_constraint (turnover) result: " << res << "\n";
    
    // Group 1 constraints
    auto group1_lower = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            for (int i = 0; i < 4; ++i) grad[i] = 1.0;
            for (int i = 4; i < 8; ++i) grad[i] = 0.0;
        }
        return x[0] + x[1] + x[2] + x[3] - 0.15;
    };
    res = nlopt_add_inequality_constraint(opt, group1_lower, nullptr, 1e-8);
    std::cout << "add_inequality_constraint (group1_lower) result: " << res << "\n";
    
    auto group1_upper = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            for (int i = 0; i < 4; ++i) grad[i] = -1.0;
            for (int i = 4; i < 8; ++i) grad[i] = 0.0;
        }
        return 0.35 - (x[0] + x[1] + x[2] + x[3]);
    };
    res = nlopt_add_inequality_constraint(opt, group1_upper, nullptr, 1e-8);
    std::cout << "add_inequality_constraint (group1_upper) result: " << res << "\n";
    
    // Asset 2 constraints
    auto asset2_lower = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            for (int i = 0; i < n; ++i) grad[i] = (i == 4) ? 1.0 : 0.0;
        }
        return x[4] - 0.15;
    };
    res = nlopt_add_inequality_constraint(opt, asset2_lower, nullptr, 1e-8);
    std::cout << "add_inequality_constraint (asset2_lower) result: " << res << "\n";
    
    auto asset2_upper = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            for (int i = 0; i < n; ++i) grad[i] = (i == 4) ? -1.0 : 0.0;
        }
        return 0.35 - x[4];
    };
    res = nlopt_add_inequality_constraint(opt, asset2_upper, nullptr, 1e-8);
    std::cout << "add_inequality_constraint (asset2_upper) result: " << res << "\n";
    
    // Asset 3 constraints
    auto asset3_lower = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            for (int i = 0; i < n; ++i) grad[i] = (i == 5) ? 1.0 : 0.0;
        }
        return x[5] - 0.15;
    };
    res = nlopt_add_inequality_constraint(opt, asset3_lower, nullptr, 1e-8);
    std::cout << "add_inequality_constraint (asset3_lower) result: " << res << "\n";
    
    auto asset3_upper = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            for (int i = 0; i < n; ++i) grad[i] = (i == 5) ? -1.0 : 0.0;
        }
        return 0.35 - x[5];
    };
    res = nlopt_add_inequality_constraint(opt, asset3_upper, nullptr, 1e-8);
    std::cout << "add_inequality_constraint (asset3_upper) result: " << res << "\n";
    
    // Group 2 constraints
    auto group2_lower = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            for (int i = 0; i < 6; ++i) grad[i] = 0.0;
            for (int i = 6; i < 8; ++i) grad[i] = 1.0;
        }
        return x[6] + x[7] - 0.15;
    };
    res = nlopt_add_inequality_constraint(opt, group2_lower, nullptr, 1e-8);
    std::cout << "add_inequality_constraint (group2_lower) result: " << res << "\n";
    
    auto group2_upper = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            for (int i = 0; i < 6; ++i) grad[i] = 0.0;
            for (int i = 6; i < 8; ++i) grad[i] = -1.0;
        }
        return 0.35 - (x[6] + x[7]);
    };
    res = nlopt_add_inequality_constraint(opt, group2_upper, nullptr, 1e-8);
    std::cout << "add_inequality_constraint (group2_upper) result: " << res << "\n";
    
    // Bounds
    std::vector<double> lb(n, 0.0);
    nlopt_set_lower_bounds(opt, lb.data());
    
    // Settings
    nlopt_set_maxeval(opt, 1000);
    nlopt_set_ftol_abs(opt, 1e-10);
    nlopt_set_ftol_rel(opt, 1e-10);
    
    // Optimize
    std::vector<double> x = prev_w;
    double minf;
    res = nlopt_optimize(opt, x.data(), &minf);
    
    std::cout << "\nOptimization result code: " << res << "\n";
    std::cout << "Objective value: " << minf << "\n";
    std::cout << "Optimal weights:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << "  w[" << i << "] = " << x[i] << "\n";
    }
    double sum = 0, turnover = 0;
    for (int i = 0; i < n; ++i) {
        sum += x[i];
        turnover += std::abs(x[i] - prev_w[i]);
    }
    std::cout << "\nSum: " << sum << "\n";
    std::cout << "Turnover: " << turnover << "\n";
    
    nlopt_destroy(opt);
    delete turnover_data;
    
    return 0;
}
