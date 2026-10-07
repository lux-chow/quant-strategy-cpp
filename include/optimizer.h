#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>

struct nlopt_opt_s;
typedef struct nlopt_opt_s* nlopt_opt;

namespace mvo {

class Optimizer {
public:
    // 优化结果
    struct Result {
        Eigen::VectorXd weights;
        double objective_value;
        int iterations;
        bool success;
        std::string message;
    };
    
    // 优化参数
    struct Params {
        int max_iter = 1000;
        double ftol_abs = 1e-10;
        double ftol_rel = 1e-10;
        double xtol_rel = 1e-10;
    };
    
    // 目标函数类型
    using ObjectiveFunc = std::function<double(const Eigen::VectorXd&)>;
    using GradientFunc = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    
    // 约束类型: g(x) >= 0
    using ConstraintFunc = std::function<double(const Eigen::VectorXd&)>;
    using ConstraintGradFunc = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    
    explicit Optimizer(const Params& params);
    ~Optimizer();
    
    // 设置目标函数
    void setObjective(ObjectiveFunc func, GradientFunc grad);
    
    // 添加不等式约束 g(x) >= 0
    void addConstraint(ConstraintFunc c_func, ConstraintGradFunc c_grad, const std::string& name = "");
    
    // 添加等式约束 h(x) = 0
    void addEqualityConstraint(ConstraintFunc c_func, ConstraintGradFunc c_grad, const std::string& name = "");
    
    // 执行优化
    Result optimize(const Eigen::VectorXd& init_weights);
    
private:
    Params params_;
    ObjectiveFunc obj_func_;
    GradientFunc obj_grad_;
    
    struct ConstraintData {
        ConstraintFunc func;
        ConstraintGradFunc grad;
        std::string name;
    };
    
    std::vector<ConstraintData> inequality_constraints_;
    std::vector<ConstraintData> equality_constraints_;
    
    nlopt_opt opt_;
};

} // namespace mvo

#endif // OPTIMIZER_H
