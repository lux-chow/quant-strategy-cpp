#include "optimizer.h"
#include <iostream>
#include <cmath>

// NLopt C API - 核心声明
extern "C" {
    typedef struct nlopt_opt_s* nlopt_opt;
    
    typedef enum {
        NLOPT_LD_LBFGS = 5,
        NLOPT_LN_COBYLA = 6,
        NLOPT_LN_NEWUOA = 9,
        NLOPT_LD_VAR1 = 11,
        NLOPT_LD_VAR2 = 12,
        NLOPT_LD_TNEWTON = 13,
        NLOPT_LD_TNEWTON_RESTART = 14,
        NLOPT_LD_TNEWTON_PRECOND = 15,
        NLOPT_LD_TNEWTON_PRECOND_RESTART = 16,
        NLOPT_LN_NELDERMEAD = 17,
        NLOPT_LN_SBPLX = 18,
        NLOPT_LN_AUGLAG = 19,
        NLOPT_LD_AUGLAG = 20,
        NLOPT_LN_AUGLAG_EQ = 21,
        NLOPT_LD_AUGLAG_EQ = 22,
        NLOPT_GN_ESCH = 23,
        NLOPT_LD_MMA = 31,
        NLOPT_LD_CCSAQ = 33,
        NLOPT_LD_SLSQP = 34
    } nlopt_algorithm;
    
    nlopt_opt nlopt_create(nlopt_algorithm algorithm, unsigned n);
    void nlopt_destroy(nlopt_opt opt);
    int nlopt_optimize(nlopt_opt opt, double* x, double* minf);
    int nlopt_set_min_objective(nlopt_opt opt, double (*f)(unsigned n, const double* x, double* gradient, void* f_data), void* f_data);
    int nlopt_add_inequality_constraint(nlopt_opt opt, double (*c)(unsigned n, const double* x, double* gradient, void* c_data), void* c_data, double tol);
    int nlopt_add_equality_constraint(nlopt_opt opt, double (*c)(unsigned n, const double* x, double* gradient, void* c_data), void* c_data, double tol);
    int nlopt_set_maxeval(nlopt_opt opt, int maxeval);
    int nlopt_set_ftol_abs(nlopt_opt opt, double tol);
    int nlopt_set_ftol_rel(nlopt_opt opt, double tol);
    int nlopt_set_xtol_rel(nlopt_opt opt, double tol);
    int nlopt_set_lower_bounds(nlopt_opt opt, const double* lb);
    int nlopt_set_verbosity(nlopt_opt opt, int verbosity);
}

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
    
    // 创建 NLopt 实例 (LD_MMA = 31，支持非线性约束)
    // MMA (Method of Moving Asymptotes) 是一种序列近似方法，
    // 适用于凸优化问题，与 SLSQP 有相似的收敛特性
    if (opt_) {
        nlopt_destroy(opt_);
    }
    opt_ = nlopt_create(NLOPT_LD_MMA, n);  // NLOPT_LD_MMA = 31
    
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
    
    // 边界约束: w >= 0
    std::vector<double> lb(n, 0.0);
    nlopt_set_lower_bounds(opt_, lb.data());
    
    // 执行优化
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = init_weights(i);
    }
    
    double minf;
    int res = nlopt_optimize(opt_, x.data(), &minf);
    
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
