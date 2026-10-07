#include "constraints.h"
#include <cmath>

namespace mvo {

std::vector<Constraint> ConstraintBuilder::build(const Eigen::VectorXd& prev_weight,
                                                  double turnover_limit,
                                                  double group_min,
                                                  double group_max) {
    std::vector<Constraint> constraints;
    
    // 等式约束: sum(w) = 1
    Constraint eq_sum;
    eq_sum.name = "sum_one";
    eq_sum.func = [](const Eigen::VectorXd& w) {
        return eqSumOne(w);
    };
    eq_sum.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        eqSumOneGrad(w, g);
    };
    constraints.push_back(eq_sum);
    
    // 不等式约束: 换手率限制
    Constraint ineq_turnover;
    ineq_turnover.name = "turnover";
    ineq_turnover.func = [&prev_weight, turnover_limit](const Eigen::VectorXd& w) {
        return ineqTurnover(w, prev_weight, turnover_limit);
    };
    ineq_turnover.grad = [&prev_weight, turnover_limit](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        ineqTurnoverGrad(w, prev_weight, turnover_limit, g);
    };
    constraints.push_back(ineq_turnover);
    
    // 品种组1 下限: w[0]+w[1]+w[2]+w[3] >= 0.15
    Constraint ineq_g1_lower;
    ineq_g1_lower.name = "group1_lower";
    ineq_g1_lower.func = [group_min](const Eigen::VectorXd& w) {
        return ineqGroup1Lower(w, group_min);
    };
    ineq_g1_lower.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        ineqGroupGrad(w, 0, 4, g);
    };
    constraints.push_back(ineq_g1_lower);
    
    // 品种组1 上限: w[0]+w[1]+w[2]+w[3] <= 0.35
    Constraint ineq_g1_upper;
    ineq_g1_upper.name = "group1_upper";
    ineq_g1_upper.func = [group_max](const Eigen::VectorXd& w) {
        return ineqGroup1Upper(w, group_max);
    };
    ineq_g1_upper.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        Eigen::VectorXd neg_g = Eigen::VectorXd::Zero(w.size());
        ineqGroupGrad(w, 0, 4, neg_g);
        g = -neg_g;
    };
    constraints.push_back(ineq_g1_upper);
    
    // 品种2 下限: w[4] >= 0.15
    Constraint ineq_a2_lower;
    ineq_a2_lower.name = "asset2_lower";
    ineq_a2_lower.func = [group_min](const Eigen::VectorXd& w) {
        return ineqAsset2Lower(w, group_min);
    };
    ineq_a2_lower.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        g.setZero(w.size());
        g(4) = 1.0;
    };
    constraints.push_back(ineq_a2_lower);
    
    // 品种2 上限: w[4] <= 0.35
    Constraint ineq_a2_upper;
    ineq_a2_upper.name = "asset2_upper";
    ineq_a2_upper.func = [group_max](const Eigen::VectorXd& w) {
        return ineqAsset2Upper(w, group_max);
    };
    ineq_a2_upper.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        g.setZero(w.size());
        g(4) = -1.0;
    };
    constraints.push_back(ineq_a2_upper);
    
    // 品种3 下限: w[5] >= 0.15
    Constraint ineq_a3_lower;
    ineq_a3_lower.name = "asset3_lower";
    ineq_a3_lower.func = [group_min](const Eigen::VectorXd& w) {
        return ineqAsset3Lower(w, group_min);
    };
    ineq_a3_lower.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        g.setZero(w.size());
        g(5) = 1.0;
    };
    constraints.push_back(ineq_a3_lower);
    
    // 品种3 上限: w[5] <= 0.35
    Constraint ineq_a3_upper;
    ineq_a3_upper.name = "asset3_upper";
    ineq_a3_upper.func = [group_max](const Eigen::VectorXd& w) {
        return ineqAsset3Upper(w, group_max);
    };
    ineq_a3_upper.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        g.setZero(w.size());
        g(5) = -1.0;
    };
    constraints.push_back(ineq_a3_upper);
    
    // 品种组2 下限: w[6]+w[7] >= 0.15
    Constraint ineq_g2_lower;
    ineq_g2_lower.name = "group2_lower";
    ineq_g2_lower.func = [group_min](const Eigen::VectorXd& w) {
        return ineqGroup2Lower(w, group_min);
    };
    ineq_g2_lower.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        ineqGroupGrad(w, 6, 8, g);
    };
    constraints.push_back(ineq_g2_lower);
    
    // 品种组2 上限: w[6]+w[7] <= 0.35
    Constraint ineq_g2_upper;
    ineq_g2_upper.name = "group2_upper";
    ineq_g2_upper.func = [group_max](const Eigen::VectorXd& w) {
        return ineqGroup2Upper(w, group_max);
    };
    ineq_g2_upper.grad = [](const Eigen::VectorXd& w, Eigen::VectorXd& g) {
        Eigen::VectorXd neg_g = Eigen::VectorXd::Zero(w.size());
        ineqGroupGrad(w, 6, 8, neg_g);
        g = -neg_g;
    };
    constraints.push_back(ineq_g2_upper);
    
    return constraints;
}

double ConstraintBuilder::eqSumOne(const Eigen::VectorXd& w) {
    return w.sum() - 1.0;
}

void ConstraintBuilder::eqSumOneGrad(const Eigen::VectorXd& w, Eigen::VectorXd& grad) {
    grad = Eigen::VectorXd::Ones(w.size());
}

double ConstraintBuilder::ineqTurnover(const Eigen::VectorXd& w, 
                                        const Eigen::VectorXd& prev_w, 
                                        double limit) {
    return limit - (w - prev_w).cwiseAbs().sum();
}

void ConstraintBuilder::ineqTurnoverGrad(const Eigen::VectorXd& w, 
                                         const Eigen::VectorXd& prev_w, 
                                         double limit, 
                                         Eigen::VectorXd& grad) {
    grad = -(w - prev_w).cwiseSign();
}

double ConstraintBuilder::ineqGroup1Lower(const Eigen::VectorXd& w, double min_val) {
    return w.segment(0, 4).sum() - min_val;
}

double ConstraintBuilder::ineqGroup1Upper(const Eigen::VectorXd& w, double max_val) {
    return max_val - w.segment(0, 4).sum();
}

double ConstraintBuilder::ineqAsset2Lower(const Eigen::VectorXd& w, double min_val) {
    return w(4) - min_val;
}

double ConstraintBuilder::ineqAsset2Upper(const Eigen::VectorXd& w, double max_val) {
    return max_val - w(4);
}

double ConstraintBuilder::ineqAsset3Lower(const Eigen::VectorXd& w, double min_val) {
    return w(5) - min_val;
}

double ConstraintBuilder::ineqAsset3Upper(const Eigen::VectorXd& w, double max_val) {
    return max_val - w(5);
}

double ConstraintBuilder::ineqGroup2Lower(const Eigen::VectorXd& w, double min_val) {
    return w.segment(6, 2).sum() - min_val;
}

double ConstraintBuilder::ineqGroup2Upper(const Eigen::VectorXd& w, double max_val) {
    return max_val - w.segment(6, 2).sum();
}

void ConstraintBuilder::ineqGroupGrad(const Eigen::VectorXd& w, int start, int end, Eigen::VectorXd& grad) {
    grad.setZero(w.size());
    for (int i = start; i < end; ++i) {
        grad(i) = 1.0;
    }
}

} // namespace mvo
