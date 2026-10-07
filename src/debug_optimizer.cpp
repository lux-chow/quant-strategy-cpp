// Debug optimizer test
#include <iostream>
#include <Eigen/Dense>
#include <nlopt.h>
#include <cmath>

// Test objective: minimize (w[0] - 0.5)^2 + (w[1] - 0.3)^2
double test_obj(unsigned n, const double* x, double* grad, void* data) {
    if (grad) {
        grad[0] = 2 * (x[0] - 0.5);
        grad[1] = 2 * (x[1] - 0.3);
    }
    return (x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.3) * (x[1] - 0.3);
}

// Constraint: w[0] + w[1] = 1
double test_eq(unsigned n, const double* x, double* grad, void* data) {
    if (grad) {
        grad[0] = 1.0;
        grad[1] = 1.0;
    }
    return x[0] + x[1] - 1.0;
}

// Constraint: w[0] >= 0.2
double test_ineq(unsigned n, const double* x, double* grad, void* data) {
    if (grad) {
        grad[0] = -1.0;  // Negative for NLopt convention (c(x) <= 0)
        grad[1] = 0.0;
    }
    return -(x[0] - 0.2);  // Negate for NLopt
}

int main() {
    std::cout << "Testing NLopt optimizer...\n";
    
    nlopt_opt opt = nlopt_create(NLOPT_LD_MMA, 2);
    
    // Set objective
    nlopt_set_min_objective(opt, test_obj, nullptr);
    
    // Set equality constraint
    nlopt_add_equality_constraint(opt, test_eq, nullptr, 1e-8);
    
    // Set inequality constraint
    nlopt_add_inequality_constraint(opt, test_ineq, nullptr, 1e-8);
    
    // Bounds
    std::vector<double> lb = {0.0, 0.0};
    nlopt_set_lower_bounds(opt, lb.data());
    
    // Settings
    nlopt_set_maxeval(opt, 1000);
    nlopt_set_ftol_abs(opt, 1e-10);
    nlopt_set_ftol_rel(opt, 1e-10);
    
    // Initial guess
    std::vector<double> x = {0.5, 0.5};
    double minf;
    
    int result = nlopt_optimize(opt, x.data(), &minf);
    
    std::cout << "Result code: " << result << "\n";
    std::cout << "Optimal x: [" << x[0] << ", " << x[1] << "]\n";
    std::cout << "Minimum value: " << minf << "\n";
    std::cout << "Expected: w[0] = 0.7, w[1] = 0.3\n";
    
    nlopt_destroy(opt);
    
    // Test with std::function approach
    std::cout << "\n\nTesting std::function approach...\n";
    
    auto obj_func = [](const Eigen::VectorXd& w) {
        return (w(0) - 0.5) * (w(0) - 0.5) + (w(1) - 0.3) * (w(1) - 0.3);
    };
    
    auto obj_grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        g(0) = 2 * (w(0) - 0.5);
        g(1) = 2 * (w(1) - 0.3);
    };
    
    struct MyData {
        decltype(obj_func)* f;
        decltype(obj_grad)* g;
    };
    
    MyData data{&obj_func, &obj_grad};
    
    auto obj_c = [](unsigned n, const double* x, double* grad, void* data) -> double {
        auto* d = static_cast<MyData*>(data);
        Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
        
        if (grad) {
            Eigen::VectorXd g(n);
            (*d->g)(w, g);
            for (unsigned i = 0; i < n; ++i) {
                grad[i] = g(i);
            }
        }
        return (*d->f)(w);
    };
    
    opt = nlopt_create(NLOPT_LD_MMA, 2);
    nlopt_set_min_objective(opt, obj_c, &data);
    nlopt_add_equality_constraint(opt, test_eq, nullptr, 1e-8);
    nlopt_add_inequality_constraint(opt, test_ineq, nullptr, 1e-8);
    nlopt_set_lower_bounds(opt, lb.data());
    nlopt_set_maxeval(opt, 1000);
    
    x = {0.5, 0.5};
    result = nlopt_optimize(opt, x.data(), &minf);
    
    std::cout << "Result code: " << result << "\n";
    std::cout << "Optimal x: [" << x[0] << ", " << x[1] << "]\n";
    std::cout << "Minimum value: " << minf << "\n";
    
    nlopt_destroy(opt);
    
    return 0;
}
