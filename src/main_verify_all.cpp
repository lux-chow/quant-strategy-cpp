// Comprehensive verification program
#include "mvo.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>

using namespace mvo;

// Read baseline file
Eigen::VectorXd loadBaseline(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return Eigen::VectorXd(0);
    
    std::vector<double> values;
    std::string line;
    std::getline(file, line);  // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::getline(ss, val, ',');
        std::getline(ss, val, ',');
        values.push_back(std::stod(val));
    }
    file.close();
    
    Eigen::VectorXd result(values.size());
    for (size_t i = 0; i < values.size(); ++i) result(i) = values[i];
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
    std::cout << "     C++ MVO Portfolio - 完整数值一致性验证\n";
    std::cout << "=============================================================\n\n";
    
    TimeSeriesData data = DataLoader::load(data_path);
    if (data.n_days == 0) {
        std::cerr << "Error: Failed to load data\n";
        return 1;
    }
    std::cout << "数据: " << data.n_days << " 天, " << data.n_assets << " 品种\n\n";
    
    // Test all 6 combinations
    struct TestCase {
        std::string name;
        Config::Strategy strategy;
        Config::CovType cov_type;
        std::string baseline_file;
    };
    
    TestCase tests[] = {
        {"maxdiv + ew",   Config::Strategy::MAXDIV, Config::CovType::EQUAL_WEIGHT,    "baseline_r_maxdiv_ew.csv"},
        {"maxdiv + exp",  Config::Strategy::MAXDIV, Config::CovType::EXPONENTIAL,     "baseline_r_maxdiv_exp.csv"},
        {"maxret + ew",   Config::Strategy::MAXRET, Config::CovType::EQUAL_WEIGHT,    "baseline_r_maxret_ew.csv"},
        {"maxret + exp",  Config::Strategy::MAXRET, Config::CovType::EXPONENTIAL,     "baseline_r_maxret_exp.csv"},
        {"risk20 + ew",   Config::Strategy::RISK20, Config::CovType::EQUAL_WEIGHT,   "baseline_r_risk20_ew.csv"},
        {"risk20 + exp",  Config::Strategy::RISK20, Config::CovType::EXPONENTIAL,    "baseline_r_risk20_exp.csv"},
    };
    
    std::cout << "=============================================================\n";
    std::cout << "                     验证结果\n";
    std::cout << "=============================================================\n";
    printf("%-20s | %-12s | %-12s | %-12s | %s\n", "策略", "C++ 最终收益", "Python 最终收益", "误差", "评级");
    std::cout << "-------------------------------------------------------------\n";
    
    for (const auto& test : tests) {
        Config config;
        config.strategy = test.strategy;
        config.cov_type = test.cov_type;
        
        auto start = std::chrono::high_resolution_clock::now();
        MVOPortfolio portfolio(config);
        portfolio.run(data);
        auto end = std::chrono::high_resolution_clock::now();
        double cpp_time = std::chrono::duration<double, std::milli>(end - start).count();
        
        Eigen::VectorXd cumulative_cpp = portfolio.getCumulativeReturns();
        Eigen::VectorXd baseline = loadBaseline(baseline_dir + "/" + test.baseline_file);
        
        if (baseline.size() == 0) {
            std::cerr << "Error: Failed to load " << test.baseline_file << "\n";
            continue;
        }
        
        int n = std::min(static_cast<int>(cumulative_cpp.size()), static_cast<int>(baseline.size()));
        
        double max_abs_err = 0.0;
        double sum_err = 0.0;
        for (int i = 0; i < n; ++i) {
            double abs_err = std::abs(cumulative_cpp(i) - baseline(i));
            max_abs_err = std::max(max_abs_err, abs_err);
            sum_err += abs_err;
        }
        
        double final_cpp = cumulative_cpp(n - 1);
        double final_py = baseline(n - 1);
        double final_err = std::abs(final_cpp - final_py);
        
        std::string grade;
        if (max_abs_err <= 1e-6) grade = "优秀";
        else if (max_abs_err <= 1e-4) grade = "良好";
        else if (max_abs_err <= 1e-3) grade = "可接受";
        else grade = "需改进";
        
        printf("%-20s | %-12.6f | %-12.6f | %-12.2e | %s\n", 
               test.name.c_str(), final_cpp, final_py, final_err, grade.c_str());
        printf("  (耗时: %.1f ms, 最大误差: %.2e)\n", cpp_time, max_abs_err);
    }
    
    std::cout << "=============================================================\n";
    
    return 0;
}
