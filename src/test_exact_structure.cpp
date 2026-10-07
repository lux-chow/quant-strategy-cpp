// Test SLSQP with actual mvo_core code structure
#include <iostream>
#include <vector>
#include <functional>
#include <Eigen/Dense>

// Simulate the exact structure of mvo_core

// CallbackData - exactly as in optimizer.cpp
struct CallbackData {
    std::function<double(const Eigen::VectorXd&)>* obj_func;
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>* obj_grad;
};

// ConstraintData - exactly as in optimizer.cpp
struct ConstraintData {
    std::function<double(const Eigen::VectorXd&)> func;
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> grad;
    std::string name;
};

// Test with nlopt
#include <nlopt.h>

int main() {
    int n = 8;
    Eigen::VectorXd init_weights(n);
    init_weights << 0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306;
    
    // Create objective functions
    std::function<double(const Eigen::VectorXd&)> obj_func;
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)> obj_grad;
    
    // MVO objective with copied data
    Eigen::VectorXd mean_ret(n);
    mean_ret.setConstant(0.01);
    Eigen::MatrixXd cov(n, n);
    cov.setIdentity() * 0.001;
    double risk_averse = 20.0;
    
    obj_func = [mean_ret, cov, risk_averse](const Eigen::VectorXd& w) {
        double ret_term = w.dot(mean_ret);
        double var_term = w.dot(cov * w);
        return -ret_term + 0.5 * risk_averse * var_term;
    };
    
    obj_grad = [mean_ret, cov, risk_averse](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        g = -mean_ret + risk_averse * (cov * w);
    };
    
    // Create constraints
    Eigen::VectorXd prev_weight = init_weights;
    std::vector<ConstraintData> constraints;
    
    // eq: sum(w) = 1
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w.sum() - 1.0; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g = Eigen::VectorXd::Ones(w.size()); },
        "sum_one"
    });
    
    // ineq: turnover
    constraints.push_back({
        [prev_weight](const Eigen::VectorXd& w) { return 0.1 - (w - prev_weight).cwiseAbs().sum(); },
        [prev_weight](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g = -(w - prev_weight).cwiseSign(); },
        "turnover"
    });
    
    // Group constraints
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w.segment(0, 4).sum() - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(0, 4).setConstant(1); },
        "g1_lower"
    });
    
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w.segment(0, 4).sum(); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(0, 4).setConstant(-1); },
        "g1_upper"
    });
    
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w(4) - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(4) = 1; },
        "a2_lower"
    });
    
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w(4); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(4) = -1; },
        "a2_upper"
    });
    
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w(5) - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(5) = 1; },
        "a3_lower"
    });
    
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w(5); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g(5) = -1; },
        "a3_upper"
    });
    
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return w.segment(6, 2).sum() - 0.15; },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(6, 2).setConstant(1); },
        "g2_lower"
    });
    
    constraints.push_back({
        [](const Eigen::VectorXd& w) { return 0.35 - w.segment(6, 2).sum(); },
        [](const Eigen::VectorXd& w, Eigen::VectorXd& g) { g.setZero(w.size()); g.segment(6, 2).setConstant(-1); },
        "g2_upper"
    });
    
    std::cout << "Testing with same structure as mvo_core...\n";
    std::cout << "Objective at init: " << obj_func(init_weights) << std::endl;
    
    // Setup NLopt
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
    std::vector<ConstraintData*> constraint_ptrs;
    for (auto& c : constraints) {
        ConstraintData* c_data = new ConstraintData{c.func, c.grad, c.name};
        constraint_ptrs.push_back(c_data);
        
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
        
        if (c.name == "sum_one") {
            r = nlopt_add_equality_constraint(opt, c_func_c, c_data, 1e-10);
        } else {
            r = nlopt_add_inequality_constraint(opt, c_func_c, c_data, 1e-8);
        }
        std::cout << "add_constraint " << c.name << ": " << r << std::endl;
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
    for (auto p : constraint_ptrs) delete p;
    
    return 0;
}
