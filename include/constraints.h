#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include <Eigen/Dense>
#include <vector>
#include <functional>

namespace mvo {

// 约束函数类型: g(w) >= 0
using ConstraintFunc = std::function<double(const Eigen::VectorXd&)>;
using ConstraintGradFunc = std::function<void(const Eigen::VectorXd&, Eigen::VectorXd&)>;

struct Constraint {
    ConstraintFunc func;
    ConstraintGradFunc grad;
    std::string name;
};

class ConstraintBuilder {
public:
    // 构建所有约束条件
    // 对应 Python get_constrain(prev_weight)
    static std::vector<Constraint> build(const Eigen::VectorXd& prev_weight,
                                         double turnover_limit = 0.1,
                                         double group_min = 0.15,
                                         double group_max = 0.35);
    
    // 约束条件列表（共11个）
    // 1. w_i >= 0 (边界约束，非显式存储)
    // 2. sum(w) = 1 (等式约束)
    // 3. 0.1 - sum(|w - prev_w|) >= 0 (换手率)
    // 4. w[0]+w[1]+w[2]+w[3] - 0.15 >= 0
    // 5. 0.35 - (w[0]+w[1]+w[2]+w[3]) >= 0
    // 6. w[4] - 0.15 >= 0
    // 7. 0.35 - w[4] >= 0
    // 8. w[5] - 0.15 >= 0
    // 9. 0.35 - w[5] >= 0
    // 10. w[6]+w[7] - 0.15 >= 0
    // 11. 0.35 - (w[6]+w[7]) >= 0
    
    // 等式约束: sum(w) - 1 = 0
    static double eqSumOne(const Eigen::VectorXd& w);
    static void eqSumOneGrad(const Eigen::VectorXd& w, Eigen::VectorXd& grad);
    
    // 不等式约束: 换手率
    static double ineqTurnover(const Eigen::VectorXd& w, const Eigen::VectorXd& prev_w, double limit);
    static void ineqTurnoverGrad(const Eigen::VectorXd& w, const Eigen::VectorXd& prev_w, double limit, Eigen::VectorXd& grad);
    
    // 不等式约束: 品种组1 (w[0:4])
    static double ineqGroup1Lower(const Eigen::VectorXd& w, double min_val);
    static double ineqGroup1Upper(const Eigen::VectorXd& w, double max_val);
    static void ineqGroupGrad(const Eigen::VectorXd& w, int start, int end, Eigen::VectorXd& grad);
    
    // 不等式约束: 品种2 (w[4])
    static double ineqAsset2Lower(const Eigen::VectorXd& w, double min_val);
    static double ineqAsset2Upper(const Eigen::VectorXd& w, double max_val);
    
    // 不等式约束: 品种3 (w[5])
    static double ineqAsset3Lower(const Eigen::VectorXd& w, double min_val);
    static double ineqAsset3Upper(const Eigen::VectorXd& w, double max_val);
    
    // 不等式约束: 品种组2 (w[6:8])
    static double ineqGroup2Lower(const Eigen::VectorXd& w, double min_val);
    static double ineqGroup2Upper(const Eigen::VectorXd& w, double max_val);
};

} // namespace mvo

#endif // CONSTRAINTS_H
