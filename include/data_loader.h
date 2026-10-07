#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <Eigen/Dense>

namespace mvo {

struct TimeSeriesData {
    std::vector<std::string> dates;        // 日期
    Eigen::MatrixXd returns;                 // 日收益率矩阵 (n_days x n_assets)
    int n_days;                             // 天数
    int n_assets;                          // 资产数量
    
    TimeSeriesData() : n_days(0), n_assets(0) {}
};

class DataLoader {
public:
    static TimeSeriesData load(const std::string& csv_path);
    static Eigen::MatrixXd computeReturns(const Eigen::MatrixXd& prices);
    
private:
    static std::vector<std::string> split(const std::string& s, char delim);
};

} // namespace mvo

#endif // DATA_LOADER_H
