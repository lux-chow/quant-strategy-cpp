// Debug program to check weights
#include "mvo.h"
#include <iostream>
#include <iomanip>

using namespace mvo;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data.csv>\n";
        return 1;
    }
    
    std::string data_path = argv[1];
    
    TimeSeriesData data = DataLoader::load(data_path);
    if (data.n_days == 0) {
        std::cerr << "Error: Failed to load data\n";
        return 1;
    }
    
    std::cout << "Data: " << data.n_days << " days, " << data.n_assets << " assets\n\n";
    
    // Test maxdiv + ew
    {
        Config config;
        config.strategy = Config::Strategy::MAXDIV;
        config.cov_type = Config::CovType::EQUAL_WEIGHT;
        
        MVOPortfolio portfolio(config);
        portfolio.run(data);
        
        const auto& weights = portfolio.getWeights();
        
        std::cout << "maxdiv + ew - Weights at key days:\n";
        std::cout << std::fixed << std::setprecision(6);
        for (int i : {0, 60, 75, 90, 120, 180, 234}) {
            if (i < weights.size()) {
                std::cout << "Day " << i << ": ";
                for (int j = 0; j < std::min(8, (int)weights[i].size()); ++j) {
                    std::cout << weights[i](j) << " ";
                }
                std::cout << " (sum=" << weights[i].sum() << ")\n";
            }
        }
        std::cout << "\n";
    }
    
    // Test maxret + ew
    {
        Config config;
        config.strategy = Config::Strategy::MAXRET;
        config.cov_type = Config::CovType::EQUAL_WEIGHT;
        
        MVOPortfolio portfolio(config);
        portfolio.run(data);
        
        const auto& weights = portfolio.getWeights();
        
        std::cout << "maxret + ew - Weights at key days:\n";
        std::cout << std::fixed << std::setprecision(6);
        for (int i : {0, 60, 75, 90, 120, 180, 234}) {
            if (i < weights.size()) {
                std::cout << "Day " << i << ": ";
                for (int j = 0; j < std::min(8, (int)weights[i].size()); ++j) {
                    std::cout << weights[i](j) << " ";
                }
                std::cout << " (sum=" << weights[i].sum() << ")\n";
            }
        }
        std::cout << "\n";
    }
    
    return 0;
}
