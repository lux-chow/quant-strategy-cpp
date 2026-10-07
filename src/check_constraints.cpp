// Check constraints at initial point
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

int main() {
    Eigen::VectorXd w(8);
    w << 0.089, 0.1025, 0.0635, 0.0645, 0.2572, 0.1655, 0.1272, 0.1306;
    
    std::cout << "Checking constraints at initial point:\n\n";
    
    // Equality: sum(w) = 1
    std::cout << "eq[0] sum(w) - 1 = " << (w.sum() - 1.0) << "\n";
    
    // Turnover (with prev_w = w)
    double turnover = (w - w).cwiseAbs().sum();
    std::cout << "ineq[0] 0.1 - turnover = " << (0.1 - turnover) << "\n";
    
    // Group 1
    std::cout << "ineq[1] group1_lower = " << (w.segment(0, 4).sum() - 0.15) << "\n";
    std::cout << "ineq[2] group1_upper = " << (0.35 - w.segment(0, 4).sum()) << "\n";
    
    // Asset 2
    std::cout << "ineq[3] asset2_lower = " << (w(4) - 0.15) << "\n";
    std::cout << "ineq[4] asset2_upper = " << (0.35 - w(4)) << "\n";
    
    // Asset 3
    std::cout << "ineq[5] asset3_lower = " << (w(5) - 0.15) << "\n";
    std::cout << "ineq[6] asset3_upper = " << (0.35 - w(5)) << "\n";
    
    // Group 2
    std::cout << "ineq[7] group2_lower = " << (w.segment(6, 2).sum() - 0.15) << "\n";
    std::cout << "ineq[8] group2_upper = " << (0.35 - w.segment(6, 2).sum()) << "\n";
    
    std::cout << "\nAll constraints should be >= 0 (except sum(w) which should be 0)\n";
    
    return 0;
}
