#include "preprocess.h"
#include <algorithm>

namespace mvo {

Eigen::MatrixXd Preprocessor::apply3Sigma(const Eigen::MatrixXd& returns, 
                                          int window, 
                                          int n_sigma) {
    int n_days = returns.rows();
    int n_assets = returns.cols();
    Eigen::MatrixXd result = returns;
    
    for (int i = window; i < n_days; ++i) {
        // 计算滑动窗口的均值和标准差
        Eigen::VectorXd mean(n_assets);
        Eigen::VectorXd std_vec(n_assets);
        
        for (int j = 0; j < n_assets; ++j) {
            double sum = 0.0, sum_sq = 0.0;
            for (int k = i - window; k < i; ++k) {
                double val = result(k, j);
                sum += val;
                sum_sq += val * val;
            }
            double n = static_cast<double>(window);
            mean(j) = sum / n;
            double variance = (sum_sq - sum * sum / n) / n;
            std_vec(j) = (variance > 0) ? std::sqrt(variance) : 1e-10;
        }
        
        // 截断极值
        for (int j = 0; j < n_assets; ++j) {
            double upper = mean(j) + n_sigma * std_vec(j);
            double lower = mean(j) - n_sigma * std_vec(j);
            result(i, j) = std::clamp(result(i, j), lower, upper);
        }
    }
    
    return result;
}

Eigen::MatrixXd Preprocessor::rollingSum(const Eigen::MatrixXd& returns, int keep) {
    int n_days = returns.rows();
    int n_assets = returns.cols();
    
    Eigen::MatrixXd result(n_days - keep + 1, n_assets);
    
    for (int i = 0; i <= n_days - keep; ++i) {
        result.row(i).setZero();
        for (int k = 0; k < keep; ++k) {
            result.row(i) += returns.row(i + k);
        }
    }
    
    return result;
}

Eigen::MatrixXd Preprocessor::getWindow(const Eigen::MatrixXd& data, int start, int end) {
    return data.block(start, 0, end - start, data.cols());
}

} // namespace mvo
