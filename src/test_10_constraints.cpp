// Test with all 10 constraints (like real code)
#include <iostream>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include <nlopt.h>

int main() {
    int n = 8;
    Eigen::VectorXd init_weights(n);
    init_weights << 0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306;
    
    Eigen::VectorXd prev_weight = init_weights;
    
    // Objective (simple quadratic)
    std::function<double(const Eigen::VectorXd&)> obj_func = [](const Eigen::VectorXd& w) {
        return (w - Eigen::VectorXd::Constant(8, 0.1)).squaredNorm();
    };
    
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> obj_grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        g = 2 * (w - Eigen::VectorXd::Constant(8, 0.1));
    };
    
    struct CallbackData {
        std::function<double(const Eigen::VectorXd&)>* obj_func;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>* obj_grad;
    };
    
    struct ConstraintData {
        std::function<double(const Eigen::VectorXd&)> func;
        std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> grad;
        bool is_equality;
    };
    
    std::vector<ConstraintData> constraints;
    
    // eq: sum(w) = 1
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w.sum() - 1.0; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g = Eigen::VectorXd::Ones(w.size()); },
        true
    });
    
    // ineq: turnover
    constraints.push_back({
        [&prev_weight](const Eigen::VectorXd& w) { return 0.1 - (w - prev_weight).cwiseAbs().sum(); },
        [&prev_weight](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g = -(w - prev_weight).cwiseSign(); },
        false
    });
    
    // Group 1 lower: w[0:4] >= 0.15
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w.segment(0, 4).sum() - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(0, 4).setConstant(1); },
        false
    });
    
    // Group 1 upper: w[0:4] <= 0.35
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w.segment(0, 4).sum(); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(0, 4).setConstant(-1); },
        false
    });
    
    // Asset 2 lower: w[4] >= 0.15
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w(4) - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(4) = 1; },
        false
    });
    
    // Asset 2 upper: w[4] <= 0.35
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w(4); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(4) = -1; },
        false
    });
    
    // Asset 3 lower: w[5] >= 0.15
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w(5) - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(5) = 1; },
        false
    });
    
    // Asset 3 upper: w[5] <= 0.35
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w(5); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(5) = -1; },
        false
    });
    
    // Group 2 lower: w[6:8] >= 0.15
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w.segment(6, 2).sum() - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(6, 2).setConstant(1); },
        false
    });
    
    // Group 2 upper: w[6:8] <= 0.35
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w.segment(6, 2).sum(); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(6, 2).setConstant(-1); },
        false
    });
    
    std::cout << "Testing with " << constraints.size() << " constraints (10 like real code)...\n";
    
    nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, n);
    
    CallbackData* cb_data = new CallbackData{&obj_func, &obj_grad};
    
    auto obj_func_c = [](unsigned n, const double* x, double* grad, void* data) -> double {
        auto* cb = static_cast<CallbackData*>(data);
        Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
        if (grad) {
            Eigen::VectorXd g(n);
            (*cb->obj_grad)(w, g);
            for (unsigned i = 0; i < n; ++i) grad[i] = g(i);
        }
        return (*cb->obj_func)(w);
    };
    
    nlopt_set_min_objective(opt, obj_func_c, cb_data);
    
    // Add constraints
    for (size_t i = 0; i < constraints.size(); ++i) {
        ConstraintData* c_data = new ConstraintData{constraints[i].func, constraints[i].grad, constraints[i].is_equality};
        
        auto c_func_c = [](unsigned n, const double* x, double* grad, void* data) -> double {
            auto* cd = static_cast<ConstraintData*>(data);
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            if (grad) {
                Eigen::VectorXd g(n);
                cd->grad(w, g);
                for (unsigned j = 0; j < n; ++j) grad[j] = g(j);
            }
            return cd->func(w);
        };
        
        int r;
        if (constraints[i].is_equality) {
            r = nlopt_add_equality_constraint(opt, c_func_c, c_data, 1e-10);
        } else {
            r = nlopt_add_inequality_constraint(opt, c_func_c, c_data, 1e-8);
        }
        std::cout << "add_constraint[" << i << "]: " << r << std::endl;
    }
    
    std::vector<double> lb(n, 0.0);
    nlopt_set_lower_bounds(opt, lb.data());
    nlopt_set_maxeval(opt, 1000);
    
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = init_weights(i);
    
    double minf;
    std::cout << "Calling nlopt_optimize..." << std::endl;
    int res = nlopt_optimize(opt, x.data(), &minf);
    std::cout << "Result: " << res << ", minf: " << minf << std::endl;
    
    nlopt_destroy(opt);
    delete cb_data;
    
    return 0;
}
