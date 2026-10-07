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
    
    // 计算协方差矩阵 (ddof=1, 与 pandas 保持一致)
    Eigen::MatrixXd centered = returns.rowwise() - mean.transpose();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / (n - 1);
    
    return cov;
}

Eigen::VectorXd CovarianceCalculator::equalWeightStd(const Eigen::MatrixXd& returns) {
    // 与 Python pandas 保持一致：ddof=1 (默认)
    // std = sqrt(sum((x - mean)^2) / (n-1))
    int n = returns.rows();
    if (n <= 1) return Eigen::VectorXd::Zero(returns.cols());
    
    // 从协方差矩阵的 diagonal 提取标准差（ddof=1 一致）
    Eigen::MatrixXd cov = equalWeight(returns);
    return cov.diagonal().cwiseSqrt();
}

Eigen::MatrixXd CovarianceCalculator::exponentialWeight(const Eigen::MatrixXd& returns, int span) {
    int n = returns.rows();
    int d = returns.cols();
    
    if (n == 0) return Eigen::MatrixXd(d, d);
    
    // alpha = 2/(span+1)，与 pandas ewm(span=span, adjust=False) 一致
    double alpha = decayFactor(span);
    double one_minus_alpha = 1.0 - alpha;
    
    // EWM 协方差（与 pandas ewm.cov() 一致）：
    // mean_t = (1-alpha) * mean_{t-1} + alpha * r_t
    // cov_t = (1-alpha) * cov_{t-1} + alpha * (r_t - mean_{t-1})^2
    Eigen::VectorXd ema = returns.row(0).transpose();
    Eigen::MatrixXd cov(d, d);
    cov.setZero();
    
    for (int i = 1; i < n; ++i) {
        Eigen::VectorXd delta = returns.row(i).transpose() - ema;
        cov = one_minus_alpha * cov + alpha * (delta * delta.transpose());
        ema = one_minus_alpha * ema + alpha * returns.row(i).transpose();
    }
    
    return cov;
}

Eigen::VectorXd CovarianceCalculator::exponentialWeightMean(const Eigen::MatrixXd& returns, int span) {
    int n = returns.rows();
    int d = returns.cols();
    
    if (n == 0) return Eigen::VectorXd(d);
    
    // alpha = 2/(span+1)，与 pandas ewm(span=span, adjust=False) 一致
    double alpha = decayFactor(span);
    double one_minus_alpha = 1.0 - alpha;
    
    // 递归计算 EMA: ema_t = (1-alpha) * ema_{t-1} + alpha * r_t
    Eigen::VectorXd ema = returns.row(0).transpose();
    for (int i = 1; i < n; ++i) {
        ema = one_minus_alpha * ema + alpha * returns.row(i).transpose();
    }
    
    return ema;
}

Eigen::VectorXd CovarianceCalculator::exponentialWeightStd(const Eigen::MatrixXd& returns, int span) {
    Eigen::MatrixXd cov = exponentialWeight(returns, span);
    return cov.diagonal().cwiseSqrt();
}

} // namespace mvo
