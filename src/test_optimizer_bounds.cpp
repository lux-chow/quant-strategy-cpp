#include "optimizer.h"
#include "objective.h"
#include "constraints.h"
#include "covariance.h"
#include "preprocess.h"
#include <iostream>
#include <Eigen/Dense>

using namespace mvo;

int main() {
    // Load data
    auto prices = load_prices_from_csv("../../data/data.csv");
    auto rets = compute_log_returns(prices);
    
    int window = 60, keep = 15, n_sigma = 3;
    auto clipped = clip_3sigma(rets, window, n_sigma);
    
    // Day 60 with EW covariance
    int day = 60;
    auto wsum = rolling_sum(clipped, keep);
    auto wsum_60 = wsum.row(day);
    
    Eigen::MatrixXd cov_full = compute_covariance_ew(clipped, day, window, 0.94);
    Eigen::VectorXd mean_ret = wsum_60 / keep;
    
    // Test optimizer A: MMA + lower bounds only
    {
        std::cout << "=== Test A: MMA + lower bounds only ===" << std::endl;
        Optimizer::Params p; p.max_iter = 1000; p.ftol_abs = 1e-9; p.ftol_rel = 1e-9; p.xtol_rel = 1e-9;
        Optimizer opt(p);
        
        Objective obj(Objective::Type::MAXDIV, mean_ret, cov_full, std_vec(cov_full), nullptr);
        opt.setObjective(obj.func(), obj.grad());
        
        ConstraintBuilder cb;
        Eigen::VectorXd prev_w = Eigen::VectorXd::Constant(8, 1.0/8);
        auto cons = cb.build(prev_w);
        for (auto& c : cons.ineq) {
            opt.addConstraint(c.func, c.grad, c.name);
        }
        for (auto& c : cons.eq) {
            opt.addEqualityConstraint(c.func, c.grad, c.name);
        }
        
        Eigen::VectorXd init = Eigen::VectorXd::Constant(8, 1.0/8);
        auto res = opt.optimize(init);
        std::cout << "Sum of weights: " << res.weights.sum() << std::endl;
        std::cout << "Success: " << res.success << std::endl;
    }
    
    // Test optimizer B: MMA + lower AND upper bounds
    {
        std::cout << "\n=== Test B: MMA + lower AND upper bounds ===" << std::endl;
        Optimizer::Params p; p.max_iter = 1000; p.ftol_abs = 1e-9; p.ftol_rel = 1e-9; p.xtol_rel = 1e-9;
        Optimizer opt(p);
        
        Objective obj(Objective::Type::MAXDIV, mean_ret, cov_full, std_vec(cov_full), nullptr);
        opt.setObjective(obj.func(), obj.grad());
        
        ConstraintBuilder cb;
        Eigen::VectorXd prev_w = Eigen::VectorXd::Constant(8, 1.0/8);
        auto cons = cb.build(prev_w);
        for (auto& c : cons.ineq) {
            opt.addConstraint(c.func, c.grad, c.name);
        }
        for (auto& c : cons.eq) {
            opt.addEqualityConstraint(c.func, c.grad, c.name);
        }
        
        Eigen::VectorXd init = Eigen::VectorXd::Constant(8, 1.0/8);
        auto res = opt.optimize(init);
        std::cout << "Sum of weights: " << res.weights.sum() << std::endl;
        std::cout << "Success: " << res.success << std::endl;
    }
    
    // Test optimizer C: SLSQP + lower bounds only
    {
        std::cout << "\n=== Test C: SLSQP + lower bounds only ===" << std::endl;
        std::cout << "(This will likely fail - SLSQP needs proper bounds)" << std::endl;
    }
    
    return 0;
}
