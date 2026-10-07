#include "mvo.h"
#include <iostream>
#include <chrono>

namespace mvo {

MVOPortfolio::MVOPortfolio(const Config& config) : config_(config) {}

void MVOPortfolio::run(const TimeSeriesData& data) {
    data_ = data;
    
    int n_days = data.n_days;
    int n_assets = data.n_assets;
    
    // 1. data_loader 已计算日收益率，直接使用
    Eigen::MatrixXd daily_returns = data.returns;
    
    // 2. 3σ 极值处理
    processed_returns_ = Preprocessor::apply3Sigma(
        daily_returns, 
        config_.window, 
        config_.n_sigma
    );
    
    // 3. 初始化权重
    Eigen::VectorXd init_weight(n_assets);
    for (int i = 0; i < n_assets; ++i) {
        init_weight(i) = Config::default_init_weight[i];
    }
    
    weights_.resize(n_days);
    weighted_returns_.resize(n_days);
    
    for (int i = 0; i < n_days; ++i) {
        weights_[i] = init_weight;
    }
    
    // 4. 每日优化
    for (int i = config_.window; i < n_days; ++i) {
        // 检查是否需要调整权重 (每 keep 天调整一次)
        if ((i - config_.window) % config_.keep == 0) {
            // 获取回看窗口的收益率
            Eigen::MatrixXd window_returns = Preprocessor::getWindow(
                processed_returns_, i - config_.window, i
            );
            
            // Rolling sum（注意：这会减少行数）
            window_returns = Preprocessor::rollingSum(window_returns, config_.keep);
            
            // 决定策略类型（使用配置中的设置）
            ObjectiveType obj_type;
            if (config_.strategy == Config::Strategy::MAXDIV) {
                obj_type = ObjectiveType::MAXDIV;
            } else if (config_.strategy == Config::Strategy::MAXRET) {
                obj_type = ObjectiveType::MAXRET;
            } else {
                obj_type = ObjectiveType::RISK20;
            }
            
            // 计算优化权重
            Eigen::VectorXd new_weight = optimizeDaily(
                window_returns,
                weights_[i - 1],
                obj_type,
                config_.cov_type
            );
            
            weights_[i] = new_weight;
        } else {
            // 延续前一天权重
            weights_[i] = weights_[i - 1];
        }
    }
    
    // 5. 计算加权收益率
    for (int i = 0; i < n_days; ++i) {
        weighted_returns_(i) = weights_[i].dot(daily_returns.row(i).transpose());
    }
}

Eigen::VectorXd MVOPortfolio::optimizeDaily(const Eigen::MatrixXd& window_returns,
                                            const Eigen::VectorXd& prev_weight,
                                            ObjectiveType obj_type,
                                            Config::CovType cov_type) {
    static int call_count = 0;
    call_count++;
    if (call_count <= 3) {
        std::cerr << "[DEBUG optimizeDaily #" << call_count << "] obj_type=" << (int)obj_type << ", prev_weight sum=" << prev_weight.sum() << std::endl;
    }
    
    int n_assets = window_returns.cols();
    
    Eigen::VectorXd mean_ret(n_assets);
    Eigen::VectorXd std_vec(n_assets);
    Eigen::MatrixXd cov(n_assets, n_assets);
    
    // 计算协方差和均值
    if (cov_type == Config::CovType::EQUAL_WEIGHT) {
        cov = CovarianceCalculator::equalWeight(window_returns);
        mean_ret = window_returns.colwise().mean();
        std_vec = CovarianceCalculator::equalWeightStd(window_returns);
    } else {
        cov = CovarianceCalculator::exponentialWeight(window_returns, config_.window);
        mean_ret = CovarianceCalculator::exponentialWeightMean(window_returns, config_.window);
        std_vec = CovarianceCalculator::exponentialWeightStd(window_returns, config_.window);
    }
    
    // 创建目标函数
    ObjectiveFunction::EvalFunc eval_func;
    ObjectiveFunction::GradFunc grad_func;
    
    double risk_averse = (obj_type == ObjectiveType::MAXRET) ? 0.0 : config_.risk_averse;
    
    ObjectiveFunction::create(obj_type, mean_ret, std_vec, cov, risk_averse,
                              eval_func, grad_func);
    
    // 创建优化器
    Optimizer::Params opt_params;
    opt_params.max_iter = config_.max_iter;
    opt_params.ftol_abs = config_.ftol_abs;
    opt_params.ftol_rel = config_.ftol_rel;
    opt_params.xtol_rel = config_.xtol_rel;
    
    Optimizer optimizer(opt_params);
    optimizer.setObjective(eval_func, grad_func);
    
    // 添加约束
    auto constraints = ConstraintBuilder::build(
        prev_weight, 
        config_.turnover_limit,
        config_.group_min,
        config_.group_max
    );
    
    for (auto& c : constraints) {
        if (c.name == "sum_one") {
            optimizer.addEqualityConstraint(c.func, c.grad, c.name);
        } else {
            optimizer.addConstraint(c.func, c.grad, c.name);
        }
    }
    
    // 执行优化
    Optimizer::Result result = optimizer.optimize(prev_weight);
    
    if (call_count <= 3) {
        std::cerr << "[DEBUG optimizeDaily #" << call_count << "] result.weights sum=" << result.weights.sum() << std::endl;
        std::cerr << "[DEBUG optimizeDaily #" << call_count << "] result.success=" << result.success << std::endl;
    }
    
    // 确保权重非负
    Eigen::VectorXd w = result.weights;
    for (int i = 0; i < n_assets; ++i) {
        if (w(i) < 0) w(i) = 0;
    }
    
    // 归一化
    double sum = w.sum();
    if (sum > 1e-10) {
        w /= sum;
    }
    
    if (call_count <= 3) {
        std::cerr << "[DEBUG optimizeDaily #" << call_count << "] final weight sum=" << w.sum() << std::endl;
    }
    
    return w;
}

Eigen::VectorXd MVOPortfolio::getCumulativeReturns() const {
    int n = weighted_returns_.size();
    Eigen::VectorXd cumulative(n + 1);
    cumulative(0) = 1.0;
    
    for (int i = 0; i < n; ++i) {
        cumulative(i + 1) = cumulative(i) * (1 + weighted_returns_(i));
    }
    
    return cumulative;
}

} // namespace mvo
