#include "optimizer.h"
#include <iostream>
#include <cmath>
#include <nlopt.h>

namespace mvo {

struct CallbackData {
    std::function<double(const Eigen::VectorXd&)>* obj_func;
    std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>* obj_grad;
};

Optimizer::Optimizer(const Params& params) : params_(params), opt_(nullptr) {}

Optimizer::~Optimizer() {
    if (opt_) {
        nlopt_destroy(opt_);
    }
}

void Optimizer::setObjective(ObjectiveFunc func, GradientFunc grad) {
    obj_func_ = func;
    obj_grad_ = grad;
}

void Optimizer::addConstraint(ConstraintFunc c_func, ConstraintGradFunc c_grad, const std::string& name) {
    ConstraintData cd;
    cd.func = c_func;
    cd.grad = c_grad;
    cd.name = name;
    inequality_constraints_.push_back(cd);
}

void Optimizer::addEqualityConstraint(ConstraintFunc c_func, ConstraintGradFunc c_grad, const std::string& name) {
    ConstraintData cd;
    cd.func = c_func;
    cd.grad = c_grad;
    cd.name = name;
    equality_constraints_.push_back(cd);
}

Optimizer::Result Optimizer::optimize(const Eigen::VectorXd& init_weights) {
    Result result;
    result.weights = init_weights;
    result.iterations = 0;
    result.success = false;
    
    int n = static_cast<int>(init_weights.size());
    
    // 创建 NLopt 实例 (NLOPT_LD_SLSQP = 39，支持非线性等式/不等式约束)
    // SLSQP 与 scipy.optimize.minimize(method='SLSQP') 等价
    if (opt_) {
        nlopt_destroy(opt_);
    }
    opt_ = nlopt_create(NLOPT_LD_SLSQP, n);
    
    // 设置目标函数
    auto* cb_data = new CallbackData{&obj_func_, &obj_grad_};
    
    auto obj_func_c = [](unsigned n, const double* x, double* grad, void* data) -> double {
        auto* cb = static_cast<CallbackData*>(data);
        Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
        
        if (grad) {
            Eigen::VectorXd g(n);
            (*cb->obj_grad)(w, g);
            for (unsigned i = 0; i < n; ++i) {
                grad[i] = g(i);
            }
        }
        return (*cb->obj_func)(w);
    };
    
    nlopt_set_min_objective(opt_, obj_func_c, cb_data);
    
    // 添加不等式约束 g(x) >= 0
    for (size_t i = 0; i < inequality_constraints_.size(); ++i) {
        auto& c = inequality_constraints_[i];
        auto* c_data = new ConstraintData{c.func, c.grad, c.name};
        
        auto c_func_c = [](unsigned n, const double* x, double* grad, void* data) -> double {
            auto* cd = static_cast<ConstraintData*>(data);
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            
            if (grad) {
                Eigen::VectorXd g(n);
                cd->grad(w, g);
                for (unsigned j = 0; j < n; ++j) {
                    grad[j] = -g(j);  // NLopt uses g(x) <= 0
                }
            }
            return -cd->func(w);  // Negate for NLopt convention
        };
        
        nlopt_add_inequality_constraint(opt_, c_func_c, c_data, 1e-8);
    }
    
    // 添加等式约束 h(x) = 0
    for (size_t i = 0; i < equality_constraints_.size(); ++i) {
        auto& c = equality_constraints_[i];
        auto* c_data = new ConstraintData{c.func, c.grad, c.name};
        
        auto eq_func_c = [](unsigned n, const double* x, double* grad, void* data) -> double {
            auto* cd = static_cast<ConstraintData*>(data);
            Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
            
            if (grad) {
                Eigen::VectorXd g(n);
                cd->grad(w, g);
                for (unsigned j = 0; j < n; ++j) {
                    grad[j] = g(j);
                }
            }
            return cd->func(w);
        };
        
        nlopt_add_equality_constraint(opt_, eq_func_c, c_data, 1e-8);
    }
    
    // 设置终止条件
    nlopt_set_maxeval(opt_, params_.max_iter);
    nlopt_set_ftol_abs(opt_, params_.ftol_abs);
    nlopt_set_ftol_rel(opt_, params_.ftol_rel);
    nlopt_set_xtol_rel(opt_, params_.xtol_rel);
    
    // 边界约束: w >= 0, w <= 1
    std::vector<double> lb(n, 0.0);
    nlopt_set_lower_bounds(opt_, lb.data());
    std::vector<double> ub(n, 1.0);
    nlopt_set_upper_bounds(opt_, ub.data());
    
    // 执行优化
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = init_weights(i);
    }
    
    double minf;
    int res = nlopt_optimize(opt_, x.data(), &minf);
    
    result.objective_value = minf;
    result.iterations = 0;  // NLopt doesn't provide iteration count
    result.success = (res > 0);  // Positive return codes are success
    
    for (int i = 0; i < n; ++i) {
        result.weights(i) = x[i];
    }
    
    // 清理回调数据
    delete cb_data;
    for (auto& c : inequality_constraints_) {
        // Note: 需要存储和删除动态分配的数据
    }
    
    return result;
}

} // namespace mvo
