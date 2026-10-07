#include "mvo.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace mvo;

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " <data.csv> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --strategy <maxdiv|maxret|risk20>  Strategy type (default: maxdiv)\n";
    std::cout << "  --cov <ew|exp>                     Covariance type (default: ew)\n";
    std::cout << "  --output <file>                     Output file for returns\n";
    std::cout << "  --weights <file>                    Output file for weights\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string data_path = argv[1];
    
    // 默认参数
    Config::Strategy strategy = Config::Strategy::MAXDIV;
    Config::CovType cov_type = Config::CovType::EQUAL_WEIGHT;
    std::string output_return = "";
    std::string output_weights = "";
    
    // 解析命令行参数
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--strategy" && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "maxdiv") strategy = Config::Strategy::MAXDIV;
            else if (s == "maxret") strategy = Config::Strategy::MAXRET;
            else if (s == "risk20") strategy = Config::Strategy::RISK20;
        } else if (arg == "--cov" && i + 1 < argc) {
            std::string c = argv[++i];
            if (c == "ew") cov_type = Config::CovType::EQUAL_WEIGHT;
            else if (c == "exp") cov_type = Config::CovType::EXPONENTIAL;
        } else if (arg == "--output" && i + 1 < argc) {
            output_return = argv[++i];
        } else if (arg == "--weights" && i + 1 < argc) {
            output_weights = argv[++i];
        }
    }
    
    // 加载数据
    auto start_load = std::chrono::high_resolution_clock::now();
    TimeSeriesData data = DataLoader::load(data_path);
    auto end_load = std::chrono::high_resolution_clock::now();
    
    if (data.n_days == 0) {
        std::cerr << "Error: Failed to load data from " << data_path << std::endl;
        return 1;
    }
    
    double load_time = std::chrono::duration<double, std::milli>(end_load - start_load).count();
    
    // 配置
    Config config;
    
    // 运行优化
    auto start_run = std::chrono::high_resolution_clock::now();
    
    MVOPortfolio portfolio(config);
    portfolio.run(data);
    
    auto end_run = std::chrono::high_resolution_clock::now();
    double run_time = std::chrono::duration<double, std::milli>(end_run - start_run).count();
    
    // 获取结果
    const auto& weights = portfolio.getWeights();
    const auto& returns = portfolio.getWeightedReturns();
    const auto& dates = portfolio.getDates();
    
    // 输出技术指标
    Eigen::VectorXd cumulative_ret = portfolio.getCumulativeReturns();
    
    std::cout << "\n策略: ";
    if (strategy == Config::Strategy::MAXDIV) std::cout << "最大分散度";
    else if (strategy == Config::Strategy::MAXRET) std::cout << "最大收益率";
    else std::cout << "风险厌恶-20";
    std::cout << "\n";
    
    std::cout << "协方差: ";
    std::cout << (cov_type == Config::CovType::EQUAL_WEIGHT ? "等权重" : "指数加权");
    std::cout << "\n\n";
    
    std::cout << "技术指标：\n";
    std::cout << Metrics::formatMetrics(returns, dates) << "\n\n";
    
    std::cout << "数据加载: " << load_time << " ms\n";
    std::cout << "优化求解: " << run_time << " ms\n";
    std::cout << "总计: " << (load_time + run_time) << " ms\n\n";
    
    // 保存结果
    if (!output_return.empty()) {
        std::ofstream ofs(output_return);
        ofs << "Date,0\n";
        ofs << std::fixed << std::setprecision(10);
        for (int i = 0; i < cumulative_ret.size(); ++i) {
            if (i < static_cast<int>(dates.size())) {
                ofs << dates[i] << "," << cumulative_ret(i) << "\n";
            }
        }
        ofs.close();
        std::cout << "收益已保存至: " << output_return << "\n";
    }
    
    if (!output_weights.empty()) {
        std::ofstream ofs(output_weights);
        ofs << "Date,a,b,c,d,e,f,g,h\n";
        ofs << std::fixed << std::setprecision(10);
        for (int i = 0; i < weights.size(); ++i) {
            ofs << dates[i];
            for (int j = 0; j < weights[i].size(); ++j) {
                ofs << "," << weights[i](j);
            }
            ofs << "\n";
        }
        ofs.close();
        std::cout << "权重已保存至: " << output_weights << "\n";
    }
    
    return 0;
}
