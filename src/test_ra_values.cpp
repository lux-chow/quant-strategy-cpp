// Test NLopt SLSQP with different risk_averse values
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <nlopt.h>
#include <cmath>

double prev_w[8] = {0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306};

// MVO objective data
struct MVOData {
    Eigen::VectorXd mean_ret;
    Eigen::VectorXd std_vec;
    Eigen::MatrixXd cov;
    double risk_averse;
};

// MVO objective: -w'μ + 0.5 * ra * w'Σw
double mvo_obj(unsigned n, const double* x, double* grad, void* data) {
    auto* d = static_cast<MVOData*>(data);
    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(x, n);
    
    if (grad) {
        Eigen::VectorXd g = -d->mean_ret + d->risk_averse * (d->cov * w);
        for (unsigned i = 0; i < n; ++i) {
            grad[i] = g(i);
        }
    }
    
    double ret_term = w.dot(d->mean_ret);
    double var_term = w.dot(d->cov * w);
    return -ret_term + 0.5 * d->risk_averse * var_term;
}

// Constraint: sum(w) = 1
double eq_sum(unsigned n, const double* x, double* grad, void* data) {
    if (grad) {
        for (int i = 0; i < n; ++i) grad[i] = 1.0;
    }
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += x[i];
    return sum - 1.0;
}

// Constraint: turnover limit
double ineq_turnover(unsigned n, const double* x, double* grad, void* data) {
    if (grad) {
        for (int i = 0; i < n; ++i) {
            grad[i] = -(x[i] > prev_w[i] ? 1.0 : -1.0);
        }
    }
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += std::abs(x[i] - prev_w[i]);
    return 0.1 - sum;
}

int main() {
    int n = 8;
    
    // Create test data
    Eigen::MatrixXd window_returns(46, 8);
    window_returns.setRandom();
    window_returns *= 0.01;
    
    Eigen::VectorXd mean = window_returns.colwise().mean();
    Eigen::MatrixXd centered = window_returns.rowwise() - mean.transpose();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / 46;
    Eigen::VectorXd std_vec = cov.diagonal().cwiseSqrt();
    
    // Test different risk_averse values
    double ra_values[] = {0.1, 0.5, 1.0, 5.0, 10.0, 20.0};
    
    for (double ra : ra_values) {
        std::cout << "=== risk_averse = " << ra << " ===\n";
        
        MVOData* data = new MVOData{mean, std_vec, cov, ra};
        
        nlopt_opt opt = nlopt_create(NLOPT_LD_SLSQP, n);
        nlopt_set_min_objective(opt, mvo_obj, data);
        nlopt_add_equality_constraint(opt, eq_sum, nullptr, 1e-10);
        nlopt_add_inequality_constraint(opt, ineq_turnover, nullptr, 1e-8);
        
        std::vector<double> lb(n, 0.0);
        nlopt_set_lower_bounds(opt, lb.data());
        nlopt_set_maxeval(opt, 1000);
        
        std::vector<double> x(8);
        for (int i = 0; i < n; ++i) x[i] = prev_w[i];
        
        double minf;
        int res = nlopt_optimize(opt, x.data(), &minf);
        
        std::cout << "Result: " << res << ", minf: " << minf;
        if (res >= 0 || res == -4) {
            std::cout << ", w[0:3]=" << x[0] << "," << x[1] << "," << x[2];
            std::cout << ", sum(w)=" << (x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]);
        }
        std::cout << "\n";
        
        nlopt_destroy(opt);
        delete data;
    }
    
    return 0;
}
