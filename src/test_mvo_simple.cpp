// Test with MVO objective
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
    
    // MVO objective data
    Eigen::VectorXd mean_ret(n);
    mean_ret.setConstant(0.01);
    Eigen::MatrixXd cov(n, n);
    cov.setIdentity() *= 0.001;
    double risk_averse = 20.0;
    
    std::function<double(const Eigen::VectorXd&)> obj_func;
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> obj_grad;
    
    obj_func = [&mean_ret, &cov, risk_averse](const Eigen::VectorXd& w) {
        double ret_term = w.dot(mean_ret);
        double var_term = w.dot(cov * w);
        return -ret_term + 0.5 * risk_averse * var_term;
    };
    
    obj_grad = [&mean_ret, &cov, risk_averse](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        g = -mean_ret + risk_averse * (cov * w);
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
    
    // Group 1 lower
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w.segment(0, 4).sum() - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(0, 4).setConstant(1); },
        false
    });
    
    // Group 1 upper
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w.segment(0, 4).sum(); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(0, 4).setConstant(-1); },
        false
    });
    
    // Asset 2
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w(4) - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(4) = 1; },
        false
    });
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w(4); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(4) = -1; },
        false
    });
    
    // Asset 3
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w(5) - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(5) = 1; },
        false
    });
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w(5); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(5) = -1; },
        false
    });
    
    // Group 2
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w.segment(6, 2).sum() - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(6, 2).setConstant(1); },
        false
    });
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w.segment(6, 2).sum(); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(6, 2).setConstant(-1); },
        false
    });
    
    std::cout << "Testing with MVO objective...\n";
    
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
    
    int r = nlopt_set_min_objective(opt, obj_func_c, cb_data);
    std::cout << "nlopt_set_min_objective: " << r << std::endl;
    
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
        
        if (constraints[i].is_equality) {
            r = nlopt_add_equality_constraint(opt, c_func_c, c_data, 1e-10);
        } else {
            r = nlopt_add_inequality_constraint(opt, c_func_c, c_data, 1e-8);
        }
        std::cout << "add_constraint[" << i << "]: " << r << std::endl;
    }
    
    std::vector<double> lb(n, 0.0);
    r = nlopt_set_lower_bounds(opt, lb.data());
    std::cout << "nlopt_set_lower_bounds: " << r << std::endl;
    
    nlopt_set_maxeval(opt, 1000);
    
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = init_weights(i);
    
    double minf;
    std::cout << "Calling nlopt_optimize..." << std::endl;
    r = nlopt_optimize(opt, x.data(), &minf);
    std::cout << "Result: " << r << ", minf: " << minf << std::endl;
    
    nlopt_destroy(opt);
    delete cb_data;
    
    return 0;
}
