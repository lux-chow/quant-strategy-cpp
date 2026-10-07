// Test to compare ObjectiveFunction::create vs manual
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
    Eigen::VectorXd mean_ret = window_sum.colwise().mean();
    
    // Initial weights
    Eigen::VectorXd init_weight(8);
    init_weight << 0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306;
    
    std::cout << std::setprecision(10);
    
    // Test 1: Manual objective function
    std::cout << "=== Manual MAXDIV objective ===\n";
    auto evalMaxDiv = [](const Eigen::VectorXd& w, const Eigen::VectorXd& std_vec, const Eigen::MatrixXd& cov) {
        double w_std = w.dot(std_vec);
        double w_cov_w = w.dot(cov * w);
        double denom = std::sqrt(w_cov_w);
        if (denom < 1e-10) return 0.0;
        return -w_std / denom;
    };
    double manual_val = evalMaxDiv(init_weight, std_vec, cov);
    std::cout << "Manual MAXDIV at init: " << manual_val << "\n";
    
    // Test 2: Using ObjectiveFunction::create
    std::cout << "\n=== ObjectiveFunction::create ===\n";
    ObjectiveFunction::EvalFunc eval;
    ObjectiveFunction::GradFunc grad;
    ObjectiveFunction::create(ObjectiveType::MAXDIV, mean_ret, std_vec, cov, 20.0, eval, grad);
    double created_val = eval(init_weight);
    std::cout << "Created MAXDIV at init: " << created_val << "\n";
    
    // Test 3: Test with different weights
    Eigen::VectorXd test_weight(8);
    test_weight << 0.15, 0.10, 0.05, 0.10, 0.25, 0.15, 0.10, 0.10;
    
    std::cout << "\n=== Test at different weights ===\n";
    double manual_test = evalMaxDiv(test_weight, std_vec, cov);
    double created_test = eval(test_weight);
    std::cout << "Manual at test weights: " << manual_test << "\n";
    std::cout << "Created at test weights: " << created_test << "\n";
    
    // Test 4: Using actual optimizer
    std::cout << "\n=== Testing optimizer ===\n";
    Optimizer::Params params;
    params.max_iter = 1000;
    Optimizer optimizer(params);
    
    // Set objective
    optimizer.setObjective(eval, grad);
    
    // Add constraints
    Eigen::VectorXd prev_weight = init_weight;
    auto constraints = ConstraintBuilder::build(prev_weight, 0.1, 0.15, 0.35);
    
    for (auto& c : constraints) {
        if (c.name == "sum_one") {
            optimizer.addEqualityConstraint(c.func, c.grad, c.name);
        } else {
            optimizer.addConstraint(c.func, c.grad, c.name);
        }
    }
    
    // Optimize
    Optimizer::Result result = optimizer.optimize(init_weight);
    
    std::cout << "Optimized weights:\n";
    for (int i = 0; i < 8; ++i) {
        std::cout << "  w[" << i << "] = " << result.weights(i) << "\n";
    }
    std::cout << "Sum: " << result.weights.sum() << "\n";
    std::cout << "Objective value: " << eval(result.weights) << "\n";
    
    return 0;
}
