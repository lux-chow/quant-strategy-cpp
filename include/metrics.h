#ifndef METRICS_H
#define METRICS_H

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace mvo {

class Metrics {
public:
    // 计算年化收益率
    static double annualizedReturn(const Eigen::VectorXd& returns, int year_days = 250);
    
    // 计算年化波动率
    static double annualizedStd(const Eigen::VectorXd& returns, int year_days = 250);
    
    // 计算夏普比率
    static double sharpeRatio(const Eigen::VectorXd& returns, int year_days = 250);
    
    // 计算最大回撤
    static double maxDrawdown(const Eigen::VectorXd& cumulative);
    
    // 计算胜率
    static double winRate(const Eigen::VectorXd& returns);
    
    // 输出技术指标字符串
    static std::string formatMetrics(const Eigen::VectorXd& returns, 
                                     const std::vector<std::string>& dates);
};

} // namespace mvo

#endif // METRICS_H
