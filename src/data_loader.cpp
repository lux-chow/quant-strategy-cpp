#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace mvo {

std::vector<std::string> DataLoader::split(const std::string& s, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

TimeSeriesData DataLoader::load(const std::string& csv_path) {
    TimeSeriesData data;
    std::ifstream file(csv_path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << csv_path << std::endl;
        return data;
    }
    
    std::string line;
    std::vector<std::vector<double>> prices;  // 净值
    std::vector<std::vector<double>> returns; // 日收益率
    
    // 跳过标题行
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        auto tokens = split(line, ',');
        if (tokens.size() < 2) continue;
        
        // 第一列是日期
        data.dates.push_back(tokens[0]);
        
        // 其余列是数值（净值）
        std::vector<double> price_row;
        for (size_t i = 1; i < tokens.size(); ++i) {
            price_row.push_back(std::stod(tokens[i]));
        }
        prices.push_back(price_row);
    }
    
    file.close();
    
    if (prices.empty()) return data;
    
    int n_days = static_cast<int>(prices.size());
    int n_assets = static_cast<int>(prices[0].size());
    
    data.n_days = n_days;
    data.n_assets = n_assets;
    
    // 计算日收益率: return[i] = price[i] / price[i-1] - 1
    data.returns = Eigen::MatrixXd(n_days, n_assets);
    
    for (int i = 0; i < n_days; ++i) {
        for (int j = 0; j < n_assets; ++j) {
            if (i == 0) {
                // 第一天收益率 = 0
                data.returns(i, j) = 0.0;
            } else {
                double prev_price = prices[i-1][j];
                double curr_price = prices[i][j];
                if (std::abs(prev_price) > 1e-10) {
                    data.returns(i, j) = curr_price / prev_price - 1.0;
                } else {
                    data.returns(i, j) = 0.0;
                }
            }
        }
    }
    
    return data;
}

} // namespace mvo
