#include "mvo.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace mvo;

// 读取 Python 基准文件
Eigen::VectorXd loadBaseline(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Cannot open: " << path << std::endl;
        return Eigen::VectorXd(0);
    }
    
    std::vector<double> values;
    std::string line;
    
    // 跳过标题
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::getline(ss, val, ',');  // 跳过日期
        std::getline(ss, val, ',');  // 读取值
        values.push_back(std::stod(val));
    }
    
    file.close();
    
    Eigen::VectorXd result(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        result(i) = values[i];
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data.csv>\n";
        return 1;
    }
    
    std::string data_path = argv[1];
    std::string baseline_dir = "/home/chow/quant/suishi-quant/data";
    
    std::cout << "=============================================================\n";
    std::cout << "         C++ MVO Portfolio - 数值一致性验证\n";
    std::cout << "=============================================================\n\n";
    
    // 加载数据
    std::cout << "加载数据...\n";
    TimeSeriesData data = DataLoader::load(data_path);
    if (data.n_days == 0) {
        std::cerr << "Error: Failed to load data\n";
        return 1;
    }
    std::cout << "  数据: " << data.n_days << " 天, " << data.n_assets << " 品种\n";
    
    // 配置
    Config config;
    
    // 运行优化
    std::cout << "\n运行优化...\n";
    auto start = std::chrono::high_resolution_clock::now();
    MVOPortfolio portfolio(config);
    portfolio.run(data);
    auto end = std::chrono::high_resolution_clock::now();
    double cpp_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    Eigen::VectorXd cumulative_cpp = portfolio.getCumulativeReturns();
    std::cout << "  C++ 耗时: " << cpp_time << " ms\n";
    
    // 加载 Python 基准
    std::string baseline_path = baseline_dir + "/baseline_r_maxdiv_ew.csv";
    std::cout << "\n加载 Python 基准...\n";
    Eigen::VectorXd baseline = loadBaseline(baseline_path);
    if (baseline.size() == 0) {
        std::cerr << "Error: Failed to load baseline\n";
        return 1;
    }
    std::cout << "  Python 基准: " << baseline_path << "\n";
    
    // 计算误差
    int n = std::min(static_cast<int>(cumulative_cpp.size()), static_cast<int>(baseline.size()));
    
    double max_abs_err = 0.0;
    double sum_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double abs_err = std::abs(cumulative_cpp(i) - baseline(i));
        max_abs_err = std::max(max_abs_err, abs_err);
        sum_err += abs_err;
    }
    double mean_err = sum_err / n;
    
    // 最终收益对比
    double final_cpp = cumulative_cpp(n - 1);
    double final_py = baseline(n - 1);
    double final_err = std::abs(final_cpp - final_py);
    
    // 评级
    std::string grade;
    if (max_abs_err <= 1e-6) grade = "优秀 (≤1e-6)";
    else if (max_abs_err <= 1e-4) grade = "良好 (≤1e-4)";
    else if (max_abs_err <= 1e-3) grade = "可接受 (≤1e-3)";
    else grade = "需改进 (>1e-3)";
    
    // 输出结果
    std::cout << "\n=============================================================\n";
    std::cout << "                    验证结果\n";
    std::cout << "=============================================================\n";
    std::cout << "最终收益 - C++:     " << std::fixed << std::setprecision(6) << final_cpp << "\n";
    std::cout << "最终收益 - Python:  " << std::fixed << std::setprecision(6) << final_py << "\n";
    std::cout << "最终收益误差:       " << std::scientific << final_err << "\n";
    std::cout << "最大绝对误差:       " << std::scientific << max_abs_err << "\n";
    std::cout << "平均误差:           " << std::scientific << mean_err << "\n";
    std::cout << "精度评级:           " << grade << "\n";
    std::cout << "=============================================================\n";
    
    return 0;
}
