#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>
#include <array>

namespace mvo {

struct Config {
    // 优化参数
    int window = 60;           // 回看时间
    int keep = 15;             // 权重保持周期
    int n_sigma = 3;           // 3σ 阈值
    int max_iter = 1000;        // 最大迭代次数
    
    // 初始权重 [a, b, c, d, e, f, g, h]
    static constexpr std::array<double, 8> default_init_weight = {
        8.9 / 100.0, 10.25 / 100.0, 6.35 / 100.0, 6.45 / 100.0,
        25.72 / 100.0, 16.55 / 100.0, 12.72 / 100.0, 13.06 / 100.0
    };
    
    // 约束参数
    double turnover_limit = 0.1;    // 换手率限制
    double group_min = 0.15;       // 品种组下限
    double group_max = 0.35;       // 品种组上限
    
    // 风险厌恶参数
    double risk_averse = 20.0;      // 默认风险厌恶系数
    
    // 求解器容差
    double ftol_abs = 1e-10;
    double ftol_rel = 1e-10;
    double xtol_rel = 1e-10;
    
    // 风险厌恶策略
    enum class Strategy {
        MAXDIV,    // 最大分散度
        MAXRET,    // 最大收益率
        RISK20     // 风险厌恶-20
    };
    
    // 协方差估计方式
    enum class CovType {
        EQUAL_WEIGHT,    // 等权重
        EXPONENTIAL      // 指数加权
    };
};

} // namespace mvo

#endif // CONFIG_H
