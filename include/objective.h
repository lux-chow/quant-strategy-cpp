#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <Eigen/Dense>
#include <functional>

namespace mvo {

enum class ObjectiveType {
    MAXDIV,    // 最大分散度
    MAXRET,    // 最大收益率（risk_averse = 0）
    RISK20     // 均值-方差（risk_averse = 20）
};

class ObjectiveFunction {
public:
    using EvalFunc = std::function<double(const Eigen::VectorXd&)>;
    using GradFunc = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;
    
    // 创建目标函数
    static void create(
        ObjectiveType type,
        const Eigen::VectorXd& mean_ret,
        const Eigen::VectorXd& std_vec,
        const Eigen::MatrixXd& cov_mat,
        double risk_averse,
        EvalFunc& eval,
        GradFunc& grad
    );
    
    // 最大分散度目标函数
    // 目标: -w'σ / sqrt(w'Σw)
    static double evalMaxDiv(const Eigen::VectorXd& w,
                             const Eigen::VectorXd& std_vec,
                             const Eigen::MatrixXd& cov);
    static void gradMaxDiv(const Eigen::VectorXd& w,
                           const Eigen::VectorXd& std_vec,
                           const Eigen::MatrixXd& cov,
                           Eigen::VectorXd& grad);
    
    // 均值-方差目标函数
    // 目标: -w'μ + 0.5 * risk_averse * w'Σw
    static double evalMVO(const Eigen::VectorXd& w,
                          const Eigen::VectorXd& mean_ret,
                          const Eigen::MatrixXd& cov,
                          double risk_averse);
    static void gradMVO(const Eigen::VectorXd& w,
                         const Eigen::VectorXd& mean_ret,
                         const Eigen::MatrixXd& cov,
                         double risk_averse,
                         Eigen::VectorXd& grad);
};

} // namespace mvo

#endif // OBJECTIVE_H
