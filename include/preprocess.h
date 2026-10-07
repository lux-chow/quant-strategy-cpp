#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <Eigen/Dense>

namespace mvo {

class Preprocessor {
public:
    // 3σ 极值处理
    // 对超过 window 期均值 ± n_sigma * std 的数据进行截断
    static Eigen::MatrixXd apply3Sigma(const Eigen::MatrixXd& returns, 
                                        int window, 
                                        int n_sigma = 3);
    
    // Rolling window 聚合
    // 对 keep 期收益进行累加
    static Eigen::MatrixXd rollingSum(const Eigen::MatrixXd& returns, int keep);
    
    // 获取 rolling window 子矩阵
    // 返回 [start:end) 的子矩阵
    static Eigen::MatrixXd getWindow(const Eigen::MatrixXd& data, 
                                      int start, int end);
};

} // namespace mvo

#endif // PREPROCESS_H
