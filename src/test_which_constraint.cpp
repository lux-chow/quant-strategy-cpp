// Debug test: which constraint causes SLSQP to fail?
#include "mvo.h"
#include <iostream>
#include <iomanip>

using namespace mvo;

int main() {
    auto data = DataLoader::load("../../data/data.csv");
    Eigen::MatrixXd processed = Preprocessor::apply3Sigma(data.returns, 60, 3);
    
    Eigen::MatrixXd window = Preprocessor::getWindow(processed, 0, 60);
    Eigen::MatrixXd window_sum = Preprocessor::rollingSum(window, 15);
    
    Eigen::MatrixXd cov = CovarianceCalculator::equalWeight(window_sum);
    Eigen::VectorXd std_vec = CovarianceCalculator::equalWeightStd(window_sum);
    Eigen::VectorXd mean_ret = window_sum.colwise().mean();
    
    Eigen::VectorXd init(8);
    init << 0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306;
    
    std::cout << std::setprecision(10);
    
    // Check constraint values at init
    auto sum_one = [](const Eigen::VectorXd& w) { return w.sum() - 1.0; };
    auto turnover = [&](const Eigen::VectorXd& w) { 
        return 0.1 - (w - init).cwiseAbs().sum(); 
    };
    
    std::cout << "Constraint values at init:\n";
    std::cout << "  sum_one: " << sum_one(init) << "\n";
    std::cout << "  turnover: " << turnover(init) << "\n";
    
    // Test 1: SLSQP with NO constraints
    std::cout << "\n=== Test 1: SLSQP + no constraints ===" << std::endl;
    {
        Optimizer::Params p; p.max_iter = 5000;
        Optimizer opt(p);
        
        ObjectiveFunction::EvalFunc eval;
        ObjectiveFunction::GradFunc grad;
        ObjectiveFunction::create(ObjectiveType::MAXDIV, mean_ret, std_vec, cov, 20.0, eval, grad);
        opt.setObjective(eval, grad);
        
        auto res = opt.optimize(init);
        std::cout << "  Result sum: " << res.weights.sum() << ", success=" << res.success << "\n";
    }
    
    // Test 2: SLSQP with sum_one only
    std::cout << "\n=== Test 2: SLSQP + sum_one only ===" << std::endl;
    {
        Optimizer::Params p; p.max_iter = 5000;
        Optimizer opt(p);
        
        ObjectiveFunction::EvalFunc eval;
        ObjectiveFunction::GradFunc grad;
        ObjectiveFunction::create(ObjectiveType::MAXDIV, mean_ret, std_vec, cov, 20.0, eval, grad);
        opt.setObjective(eval, grad);
        
        opt.addEqualityConstraint(sum_one, 
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g = Eigen::VectorXd::Ones(8); }, "sum_one");
        
        auto res = opt.optimize(init);
        std::cout << "  Result sum: " << res.weights.sum() << ", success=" << res.success << "\n";
    }
    
    // Test 3: SLSQP with sum_one + turnover
    std::cout << "\n=== Test 3: SLSQP + sum_one + turnover ===" << std::endl;
    {
        Optimizer::Params p; p.max_iter = 5000;
        Optimizer opt(p);
        
        ObjectiveFunction::EvalFunc eval;
        ObjectiveFunction::GradFunc grad;
        ObjectiveFunction::create(ObjectiveType::MAXDIV, mean_ret, std_vec, cov, 20.0, eval, grad);
        opt.setObjective(eval, grad);
        
        opt.addEqualityConstraint(sum_one, 
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g = Eigen::VectorXd::Ones(8); }, "sum_one");
        
        Eigen::VectorXd prev = init;
        opt.addConstraint(
            [prev](const Eigen::VectorXd& w){ return 0.1 - (w - prev).cwiseAbs().sum(); },
            [prev](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g = -(w - prev).cwiseSign(); },
            "turnover");
        
        auto res = opt.optimize(init);
        std::cout << "  Result sum: " << res.weights.sum() << ", success=" << res.success << "\n";
    }
    
    // Test 4: SLSQP with all linear constraints (no turnover)
    std::cout << "\n=== Test 4: SLSQP + all linear (no turnover) ===" << std::endl;
    {
        Optimizer::Params p; p.max_iter = 5000;
        Optimizer opt(p);
        
        ObjectiveFunction::EvalFunc eval;
        ObjectiveFunction::GradFunc grad;
        ObjectiveFunction::create(ObjectiveType::MAXDIV, mean_ret, std_vec, cov, 20.0, eval, grad);
        opt.setObjective(eval, grad);
        
        opt.addEqualityConstraint(sum_one, 
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g = Eigen::VectorXd::Ones(8); }, "sum_one");
        opt.addConstraint([](const Eigen::VectorXd& w){ return w.segment(0,4).sum() - 0.15; },
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g.setZero(8); g.head(4).setOnes(); }, "g1_l");
        opt.addConstraint([](const Eigen::VectorXd& w){ return 0.35 - w.segment(0,4).sum(); },
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g.setZero(8); g.head(4).setOnes(); }, "g1_u");
        opt.addConstraint([](const Eigen::VectorXd& w){ return w(4) - 0.15; },
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g.setZero(8); g(4)=1; }, "a2_l");
        opt.addConstraint([](const Eigen::VectorXd& w){ return 0.35 - w(4); },
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g.setZero(8); g(4)=-1; }, "a2_u");
        opt.addConstraint([](const Eigen::VectorXd& w){ return w(5) - 0.15; },
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g.setZero(8); g(5)=1; }, "a3_l");
        opt.addConstraint([](const Eigen::VectorXd& w){ return 0.35 - w(5); },
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g.setZero(8); g(5)=-1; }, "a3_u");
        opt.addConstraint([](const Eigen::VectorXd& w){ return w.segment(6,2).sum() - 0.15; },
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g.setZero(8); g.tail(2).setOnes(); }, "g2_l");
        opt.addConstraint([](const Eigen::VectorXd& w){ return 0.35 - w.segment(6,2).sum(); },
            [](const Eigen::VectorXd& w, Eigen::VectorXd& g){ g.setZero(8); g.tail(2).setOnes(); }, "g2_u");
        
        auto res = opt.optimize(init);
        std::cout << "  Result sum: " << res.weights.sum() << ", success=" << res.success << "\n";
    }
    
    // Test 5: Full constraints via ConstraintBuilder
    std::cout << "\n=== Test 5: SLSQP + full constraints via ConstraintBuilder ===" << std::endl;
    {
        Optimizer::Params p; p.max_iter = 5000;
        Optimizer opt(p);
        
        ObjectiveFunction::EvalFunc eval;
        ObjectiveFunction::GradFunc grad;
        ObjectiveFunction::create(ObjectiveType::MAXDIV, mean_ret, std_vec, cov, 20.0, eval, grad);
        opt.setObjective(eval, grad);
        
        auto constraints = ConstraintBuilder::build(init, 0.1, 0.15, 0.35);
        for (auto& c : constraints) {
            if (c.name == "sum_one") {
                opt.addEqualityConstraint(c.func, c.grad, c.name);
            } else {
                opt.addConstraint(c.func, c.grad, c.name);
            }
        }
        
        auto res = opt.optimize(init);
        std::cout << "  Result sum: " << res.weights.sum() << ", success=" << res.success << "\n";
    }
    
    return 0;
}
