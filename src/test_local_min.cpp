// Check if initial point is a local minimum
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
    
    // Apply 3-sigma preprocessing
    Eigen::MatrixXd processed = Preprocessor::apply3Sigma(data.returns, 60, 3);
    
    // Get window at day 60
    Eigen::MatrixXd window = Preprocessor::getWindow(processed, 0, 60);
    Eigen::MatrixXd window_sum = Preprocessor::rollingSum(window, 15);
    
    // Calculate cov
    Eigen::MatrixXd cov = CovarianceCalculator::equalWeight(window_sum);
    Eigen::VectorXd std_vec = CovarianceCalculator::equalWeightStd(window_sum);
    
    // Initial weights
    Eigen::VectorXd init_weight(8);
    init_weight << 0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306;
    
    // Calculate maxdiv objective at init
    double w_std = init_weight.dot(std_vec);
    double w_cov_w = init_weight.dot(cov * init_weight);
    double denom = std::sqrt(w_cov_w);
    double maxdiv_init = -w_std / denom;
    
    std::cout << std::setprecision(10);
    std::cout << "MAXDIV objective at init: " << maxdiv_init << "\n\n";
    
    // Test a few alternative weights
    Eigen::VectorXd alt_weights(8);
    
    // Alt 1: Python baseline at day 60
    alt_weights << 0.10406829611300628, 0.0525484593876503, 0.06352097642538419, 0.0645034159929435, 
                   0.29210738118718815, 0.16549994970798731, 0.1271999769376189, 0.1305515442482214;
    w_std = alt_weights.dot(std_vec);
    w_cov_w = alt_weights.dot(cov * alt_weights);
    denom = std::sqrt(w_cov_w);
    double maxdiv_alt1 = -w_std / denom;
    std::cout << "MAXDIV objective at alt1 (Python): " << maxdiv_alt1 << "\n";
    
    // Alt 2: Equal weights
    alt_weights.setConstant(1.0 / 8.0);
    w_std = alt_weights.dot(std_vec);
    w_cov_w = alt_weights.dot(cov * alt_weights);
    denom = std::sqrt(w_cov_w);
    double maxdiv_alt2 = -w_std / denom;
    std::cout << "MAXDIV objective at alt2 (equal): " << maxdiv_alt2 << "\n";
    
    // Alt 3: Random perturbation
    alt_weights = init_weight + Eigen::VectorXd::Random(8) * 0.01;
    alt_weights = alt_weights.cwiseMax(0.01);  // Ensure positive
    alt_weights /= alt_weights.sum();  // Normalize
    w_std = alt_weights.dot(std_vec);
    w_cov_w = alt_weights.dot(cov * alt_weights);
    denom = std::sqrt(w_cov_w);
    double maxdiv_alt3 = -w_std / denom;
    std::cout << "MAXDIV objective at alt3 (perturbed): " << maxdiv_alt3 << "\n";
    
    std::cout << "\nIs init a local minimum? Compare values:\n";
    std::cout << "init = " << maxdiv_init << "\n";
    std::cout << "alt1 = " << maxdiv_alt1 << " (init < alt1? " << (maxdiv_init < maxdiv_alt1) << ")\n";
    std::cout << "alt2 = " << maxdiv_alt2 << " (init < alt2? " << (maxdiv_init < maxdiv_alt2) << ")\n";
    std::cout << "alt3 = " << maxdiv_alt3 << " (init < alt3? " << (maxdiv_init < maxdiv_alt3) << ")\n";
    
    return 0;
}
