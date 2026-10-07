#ifndef MVO_H
#define MVO_H

#include "config.h"
#include "data_loader.h"
#include "preprocess.h"
#include "covariance.h"
#include "objective.h"
#include "constraints.h"
#include "optimizer.h"
#include "metrics.h"

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace mvo {

class MVOPortfolio {
public:
    MVOPortfolio(const Config& config);
    
    // 运行优化
    void run(const TimeSeriesData& data);
    
    // 获取权重序列
    const std::vector<Eigen::VectorXd>& getWeights() const { return weights_; }
    
    // 获取加权收益率序列
    const Eigen::VectorXd& getWeightedReturns() const { return weighted_returns_; }
    
    // 获取累计收益序列
    Eigen::VectorXd getCumulativeReturns() const;
    
    // 获取日期列表
    const std::vector<std::string>& getDates() const { return data_.dates; }
    
    // 获取配置
    const Config& getConfig() const { return config_; }
    
private:
    Config config_;
    TimeSeriesData data_;
    
    std::vector<Eigen::VectorXd> weights_;           // 权重序列
    Eigen::VectorXd weighted_returns_;                // 加权收益率序列
    
    // 预处理后的收益率
    Eigen::MatrixXd processed_returns_;
    
    // 每日优化
    Eigen::VectorXd optimizeDaily(const Eigen::MatrixXd& window_returns,
                                  const Eigen::VectorXd& prev_weight,
                                  ObjectiveType obj_type,
                                  Config::CovType cov_type);
    
    // 计算等权重协方差
    void computeEqualWeightCov(const Eigen::MatrixXd& returns,
                               Eigen::VectorXd& mean_ret,
                               Eigen::VectorXd& std_vec,
                               Eigen::MatrixXd& cov);
    
    // 计算指数加权协方差
    void computeExpWeightCov(const Eigen::MatrixXd& returns,
                             Eigen::VectorXd& mean_ret,
                             Eigen::VectorXd& std_vec,
                             Eigen::MatrixXd& cov);
};

} // namespace mvo

#endif // MVO_H
