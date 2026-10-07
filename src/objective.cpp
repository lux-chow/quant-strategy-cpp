#include "objective.h"
#include <iostream>
#include <cmath>

namespace mvo {

void ObjectiveFunction::create(
    ObjectiveType type,
    const Eigen::VectorXd& mean_ret,
    const Eigen::VectorXd& std_vec,
    const Eigen::MatrixXd& cov_mat,
    double risk_averse,
    EvalFunc& eval,
    GradFunc& grad
) {
    // 按值拷贝 Eigen 对象，避免引用悬空
    Eigen::VectorXd mean_copy = mean_ret;
    Eigen::VectorXd std_copy = std_vec;
    Eigen::MatrixXd cov_copy = cov_mat;
    double ra_copy = risk_averse;
    
    if (type == ObjectiveType::MAXDIV) {
        eval = [std_copy, cov_copy](const Eigen::VectorXd& w) {
            return evalMaxDiv(w, std_copy, cov_copy);
        };
        grad = [std_copy, cov_copy](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
            gradMaxDiv(w, std_copy, cov_copy, g);
        };
    } else {
        // MAXRET 和 RISK20 都使用 MVO 目标函数
        // MAXRET: risk_averse = 0
        // RISK20: risk_averse = 20
        eval = [mean_copy, cov_copy, ra_copy](const Eigen::VectorXd& w) {
            return evalMVO(w, mean_copy, cov_copy, ra_copy);
        };
        grad = [mean_copy, cov_copy, ra_copy](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
            gradMVO(w, mean_copy, cov_copy, ra_copy, g);
        };
    }
}

double ObjectiveFunction::evalMaxDiv(const Eigen::VectorXd& w,
                                     const Eigen::VectorXd& std_vec,
                                     const Eigen::MatrixXd& cov) {
    // 分散度 = w'σ / sqrt(w'Σw)
    // 目标函数取负（最小化问题）
    double w_std = w.dot(std_vec);
    double w_cov_w = w.dot(cov * w);
    double denom = std::sqrt(w_cov_w);
    
    if (denom < 1e-10) return 0.0;
    
    return -w_std / denom;
}

void ObjectiveFunction::gradMaxDiv(const Eigen::VectorXd& w,
                                   const Eigen::VectorXd& std_vec,
                                   const Eigen::MatrixXd& cov,
                                   Eigen::VectorXd& grad) {
    double w_std = w.dot(std_vec);
    double w_cov_w = w.dot(cov * w);
    double denom = std::sqrt(w_cov_w);
    double denom_cubed = denom * denom * denom;
    
    if (denom < 1e-10) {
        grad.setZero();
        return;
    }
    
    // 梯度推导：
    // d/dw [-w'σ / sqrt(w'Σw)] = -σ / sqrt(w'Σw) + (w'σ / (w'Σw)^(3/2)) * Σw
    
    grad = -std_vec / denom + (w_std / denom_cubed) * (cov * w);
}

double ObjectiveFunction::evalMVO(const Eigen::VectorXd& w,
                                  const Eigen::VectorXd& mean_ret,
                                  const Eigen::MatrixXd& cov,
                                  double risk_averse) {
    // 目标函数 = -w'μ + 0.5 * risk_averse * w'Σw
    // 取负（最小化问题）
    double ret_term = w.dot(mean_ret);
    double var_term = w.dot(cov * w);
    
    return -ret_term + 0.5 * risk_averse * var_term;
}

void ObjectiveFunction::gradMVO(const Eigen::VectorXd& w,
                                const Eigen::VectorXd& mean_ret,
                                const Eigen::MatrixXd& cov,
                                double risk_averse,
                                Eigen::VectorXd& grad) {
    // 梯度 = -μ + risk_averse * Σw
    grad = -mean_ret + risk_averse * (cov * w);
}

} // namespace mvo
