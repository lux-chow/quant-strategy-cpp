#include "metrics.h"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace mvo {

double Metrics::annualizedReturn(const Eigen::VectorXd& returns, int year_days) {
    double total_return = returns.sum();
    int n = returns.size();
    double years = static_cast<double>(n) / year_days;
    
    if (years < 1e-10) return 0.0;
    
    return total_return / years;
}

double Metrics::annualizedStd(const Eigen::VectorXd& returns, int year_days) {
    int n = returns.size();
    if (n < 2) return 0.0;
    
    double mean = returns.mean();
    double variance = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = returns(i) - mean;
        variance += diff * diff;
    }
    variance /= (n - 1);
    
    double years = static_cast<double>(n) / year_days;
    return std::sqrt(variance * year_days);
}

double Metrics::sharpeRatio(const Eigen::VectorXd& returns, int year_days) {
    double ann_ret = annualizedReturn(returns, year_days);
    double ann_std = annualizedStd(returns, year_days);
    
    if (std::abs(ann_std) < 1e-10) return 0.0;
    
    return ann_ret / ann_std;
}

double Metrics::maxDrawdown(const Eigen::VectorXd& cumulative) {
    int n = cumulative.size();
    if (n < 2) return 0.0;
    
    double max_dd = 0.0;
    double peak = cumulative(0);
    
    for (int i = 1; i < n; ++i) {
        if (cumulative(i) > peak) {
            peak = cumulative(i);
        }
        double dd = peak - cumulative(i);
        if (dd > max_dd) {
            max_dd = dd;
        }
    }
    
    return max_dd;
}

double Metrics::winRate(const Eigen::VectorXd& returns) {
    int n = returns.size();
    if (n < 2) return 0.0;
    
    int wins = 0;
    for (int i = 0; i < n; ++i) {
        if (returns(i) > 0) wins++;
    }
    
    return static_cast<double>(wins) / n;
}

std::string Metrics::formatMetrics(const Eigen::VectorXd& returns, 
                                   const std::vector<std::string>& dates) {
    std::ostringstream oss;
    
    // 计算累计收益
    Eigen::VectorXd cumulative = Eigen::VectorXd::Zero(returns.size() + 1);
    cumulative(0) = 1.0;
    for (int i = 0; i < returns.size(); ++i) {
        cumulative(i + 1) = cumulative(i) * (1 + returns(i));
    }
    
    double ann_ret = annualizedReturn(returns) * 100;
    double ann_std = annualizedStd(returns) * 100;
    double sharpe = sharpeRatio(returns);
    double max_dd = maxDrawdown(cumulative) * 100;
    double win_rate = winRate(returns) * 100;
    
    oss.precision(2);
    oss << std::fixed;
    oss << ann_ret << " & " << ann_std << " & " << sharpe << " & " 
        << max_dd << " & " << win_rate;
    
    return oss.str();
}

} // namespace mvo
