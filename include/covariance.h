#ifndef COVARIANCE_H
#define COVARIANCE_H

#include <Eigen/Dense>

namespace mvo {

class CovarianceCalculator {
public:
    // 等权重协方差矩阵
    // 对应 Python: df_return.cov().values
    static Eigen::MatrixXd equalWeight(const Eigen::MatrixXd& returns);
    
    // 等权重标准差向量
    // 对应 Python: w_return.std().values
    static Eigen::VectorXd equalWeightStd(const Eigen::MatrixXd& returns);
    
    // 指数加权协方差矩阵
    // 对应 Python: w_return.ewm(span=window, adjust=False).cov().values[-len:]
    static Eigen::MatrixXd exponentialWeight(const Eigen::MatrixXd& returns, int span);
    
    // 指数加权均值向量
    // 对应 Python: w_return.ewm(span=window, adjust=False).mean().iloc[-1].values
    static Eigen::VectorXd exponentialWeightMean(const Eigen::MatrixXd& returns, int span);
    
    // 指数加权标准差向量（从协方差矩阵对角线取平方根）
    static Eigen::VectorXd exponentialWeightStd(const Eigen::MatrixXd& returns, int span);
    
private:
    // 计算衰减因子: lambda = 2 / (span + 1)
    static double decayFactor(int span);
};

} // namespace mvo

#endif // COVARIANCE_H
