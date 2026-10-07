// Test to compare objective values
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
    
    // Apply 3-sigma preprocessing
    Eigen::MatrixXd processed = Preprocessor::apply3Sigma(data.returns, 60, 3);
    
    // Get window at day 60
    Eigen::MatrixXd window = Preprocessor::getWindow(processed, 0, 60);
    Eigen::MatrixXd window_sum = Preprocessor::rollingSum(window, 15);
    
    std::cout << "Window shape: " << window.rows() << " x " << window.cols() << "\n";
    std::cout << "Window_sum shape: " << window_sum.rows() << " x " << window_sum.cols() << "\n\n";
    
    // Calculate cov
    Eigen::MatrixXd cov = CovarianceCalculator::equalWeight(window_sum);
    Eigen::VectorXd std_vec = CovarianceCalculator::equalWeightStd(window_sum);
    Eigen::VectorXd mean_ret = window_sum.colwise().mean();
    
    std::cout << "Mean returns:\n" << mean_ret.transpose() << "\n\n";
    std::cout << "Std vector:\n" << std_vec.transpose() << "\n\n";
    
    // Initial weights
    Eigen::VectorXd init_weight(8);
    init_weight << 0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306;
    
    // Calculate maxdiv objective at init weights
    double w_std = init_weight.dot(std_vec);
    double w_cov_w = init_weight.dot(cov * init_weight);
    double denom = std::sqrt(w_cov_w);
    double maxdiv_obj = -w_std / denom;
    
    std::cout << "Maxdiv objective at init weights: " << maxdiv_obj << "\n";
    std::cout << "  w_std = " << w_std << "\n";
    std::cout << "  w_cov_w = " << w_cov_w << "\n";
    std::cout << "  denom = " << denom << "\n\n";
    
    // Calculate MVO objective at init weights
    double ret_term = init_weight.dot(mean_ret);
    double var_term = init_weight.dot(cov * init_weight);
    double mvo_obj_ra0 = -ret_term;
    double mvo_obj_ra20 = -ret_term + 0.5 * 20 * var_term;
    
    std::cout << "MVO objective (ra=0) at init weights: " << mvo_obj_ra0 << "\n";
    std::cout << "MVO objective (ra=20) at init weights: " << mvo_obj_ra20 << "\n";
    std::cout << "  ret_term = " << ret_term << "\n";
    std::cout << "  var_term = " << var_term << "\n\n";
    
    // Check sum of weights
    std::cout << "Sum of init weights: " << init_weight.sum() << "\n";
    
    return 0;
}
