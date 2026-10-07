// Test NLopt constraint handling
#include <iostream>
#include <vector>
#include <nlopt.h>
#include <cmath>

int main() {
    std::cout << "Testing NLopt equality constraint handling...\n\n";
    
    // Simple problem: minimize (w0 - 0.3)^2 + (w1 - 0.7)^2
    // subject to: w0 + w1 = 1, w0 >= 0, w1 >= 0
    
    int n = 2;
    
    auto obj_func = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            grad[0] = 2 * (x[0] - 0.3);
            grad[1] = 2 * (x[1] - 0.7);
        }
        return (x[0] - 0.3) * (x[0] - 0.3) + (x[1] - 0.7) * (x[1] - 0.7);
    };
    
    auto eq_func = [](unsigned n, const double* x, double* grad, void* data) -> double {
        if (grad) {
            grad[0] = 1.0;
            grad[1] = 1.0;
        }
        return x[0] + x[1] - 1.0;
    };
    
    // Test MMA
    std::cout << "=== Testing NLOPT_LD_MMA ===\n";
    {
        nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, n);
        nlopt_set_min_objective(opt, obj_func, nullptr);
        nlopt_add_equality_constraint(opt, eq_func, nullptr, 1e-10);
        
        std::vector<double> lb = {0.0, 0.0};
        nlopt_set_lower_bounds(opt, lb.data());
        nlopt_set_maxeval(opt, 1000);
        
        std::vector<double> x = {0.5, 0.5};
        double minf;
        int result = nlopt_optimize(opt, x.data(), &minf);
        
        std::cout << "MMA Result: " << result << "\n";
        std::cout << "Solution: [" << x[0] << ", " << x[1] << "]\n";
        std::cout << "Sum: " << (x[0] + x[1]) << "\n";
        std::cout << "Expected: [0.3, 0.7], sum = 1.0\n\n";
        
        nlopt_destroy(opt);
    }
    
    // Test SLSQP
    std::cout << "=== Testing NLOPT_LD_SLSQP ===\n";
    {
        nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, n);
        nlopt_set_min_objective(opt, obj_func, nullptr);
        nlopt_add_equality_constraint(opt, eq_func, nullptr, 1e-10);
        
        std::vector<double> lb = {0.0, 0.0};
        nlopt_set_lower_bounds(opt, lb.data());
        nlopt_set_maxeval(opt, 1000);
        
        std::vector<double> x = {0.5, 0.5};
        double minf;
        int result = nlopt_optimize(opt, x.data(), &minf);
        
        std::cout << "SLSQP Result: " << result << "\n";
        std::cout << "Solution: [" << x[0] << ", " << x[1] << "]\n";
        std::cout << "Sum: " << (x[0] + x[1]) << "\n";
        std::cout << "Expected: [0.3, 0.7], sum = 1.0\n\n";
        
        nlopt_destroy(opt);
    }
    
    // Test with inequality constraint instead of equality
    std::cout << "=== Testing MMA with 2 equality constraints ===\n";
    {
        nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, n);
        nlopt_set_min_objective(opt, obj_func, nullptr);
        
        // Add equality constraint: w0 + w1 = 1
        nlopt_add_equality_constraint(opt, eq_func, nullptr, 1e-10);
        
        std::vector<double> lb = {0.0, 0.0};
        nlopt_set_lower_bounds(opt, lb.data());
        nlopt_set_maxeval(opt, 1000);
        nlopt_set_ftol_abs(opt, 1e-12);
        nlopt_set_ftol_rel(opt, 1e-12);
        
        std::vector<double> x = {0.5, 0.5};
        double minf;
        int result = nlopt_optimize(opt, x.data(), &minf);
        
        std::cout << "Result: " << result << "\n";
        std::cout << "Solution: [" << x[0] << ", " << x[1] << "]\n";
        std::cout << "Sum: " << (x[0] + x[1]) << "\n";
        
        nlopt_destroy(opt);
    }
    
    return 0;
}
