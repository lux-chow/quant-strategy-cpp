#include "covariance.h"
#include <cmath>

namespace mvo {

double CovarianceCalculator::decayFactor(int span) {
    return 2.0 / (span + 1);
}

Eigen::MatrixXd CovarianceCalculator::equalWeight(const Eigen::MatrixXd& returns) {
    int n = returns.rows();
    int d = returns.cols();
    
    // 计算均值
    Eigen::VectorXd mean = returns.colwise().mean();
    
    // 计算协方差矩阵 (ddof=0, 除以 n)
    Eigen::MatrixXd centered = returns.rowwise() - mean.transpose();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / n;
    
    return cov;
}

Eigen::VectorXd CovarianceCalculator::equalWeightStd(const Eigen::MatrixXd& returns) {
    return returns.colwise().stableNorm() / std::sqrt(returns.rows());
}

Eigen::MatrixXd CovarianceCalculator::exponentialWeight(const Eigen::MatrixXd& returns, int span) {
    int n = returns.rows();
    int d = returns.cols();
    
    if (n == 0) return Eigen::MatrixXd(d, d);
    
    double lambda = decayFactor(span);
    double one_minus_lambda = 1.0 - lambda;
    
    // 递归计算 EWM 协方差
    // cov_t = (1-lambda) * sum_{i=0}^{inf} lambda^i * (r_{t-i} - mu)'(r_{t-i} - mu)
    // 简化为从第一个数据点开始递推
    
    // 计算初始均值（使用 EWM 递归公式）
    Eigen::VectorXd ema_mean = returns.row(0);
    for (int i = 1; i < n; ++i) {
        ema_mean = one_minus_lambda * returns.row(i).transpose() + lambda * ema_mean;
    }
    
    // 计算 EWM 协方差
    Eigen::MatrixXd cov(d, d);
    cov.setZero();
    
    Eigen::VectorXd current_mean = returns.row(0).transpose();
    double alpha_sum = 1.0;
    
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd diff = returns.row(i).transpose() - current_mean;
        double weight = std::pow(lambda, n - 1 - i);
        cov += weight * diff * diff.transpose();
        alpha_sum += weight;
    }
    
    // 归一化
    // 注意：这里使用简单的递推公式，与 pandas ewm 行为可能有细微差异
    // pandas ewm.cov() 使用递归展开
    
    // 重新实现：使用展开的递归形式
    cov.setZero();
    current_mean.setZero();
    Eigen::MatrixXd cov_tmp(d, d);
    cov_tmp.setZero();
    
    double ewma_var_sum = 0.0;
    
    for (int i = 0; i < n; ++i) {
        double weight = std::pow(lambda, n - 1 - i);
        
        if (i == 0) {
            current_mean = returns.row(i).transpose();
        } else {
            // 更新 EMA 均值
            Eigen::VectorXd new_mean = one_minus_lambda * returns.row(i).transpose() + lambda * current_mean;
            
            // 更新协方差
            for (int j = 0; j < d; ++j) {
                for (int k = 0; k < d; ++k) {
                    double delta1 = returns(i, j) - current_mean(j);
                    double delta2 = returns(i, k) - current_mean(k);
                    cov(j, k) = (one_minus_lambda * cov(j, k) + 
                                 lambda * delta1 * delta2) / (one_minus_lambda * ewma_var_sum + lambda);
                }
            }
            ewma_var_sum = one_minus_lambda * ewma_var_sum + lambda;
            
            current_mean = new_mean;
        }
    }
    
    // 最终的协方差矩阵（取最后一个）
    // 返回完整的 EWM 协方差矩阵
    return cov / (1.0 - std::pow(lambda, n));
}

Eigen::VectorXd CovarianceCalculator::exponentialWeightMean(const Eigen::MatrixXd& returns, int span) {
    int n = returns.rows();
    int d = returns.cols();
    
    if (n == 0) return Eigen::VectorXd(d);
    
    double lambda = decayFactor(span);
    double one_minus_lambda = 1.0 - lambda;
    
    // 递归计算 EMA
    Eigen::VectorXd ema = returns.row(0).transpose();
    for (int i = 1; i < n; ++i) {
        ema = one_minus_lambda * returns.row(i).transpose() + lambda * ema;
    }
    
    return ema;
}

Eigen::VectorXd CovarianceCalculator::exponentialWeightStd(const Eigen::MatrixXd& returns, int span) {
    Eigen::MatrixXd cov = exponentialWeight(returns, span);
    return cov.diagonal().cwiseSqrt();
}

} // namespace mvo
