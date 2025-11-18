// In the name of Allah

#include <Activation_Functions.h>

Eigen::MatrixXd ActivationFunctions::relu_activation(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd result = input.cwiseMax(0.0) ;
    return result ;
}

Eigen::MatrixXd ActivationFunctions::softmax_activation(const Eigen::MatrixXd &input) {
    // Step 1: exponentiate every element
    Eigen::MatrixXd numerator = input.array().exp().matrix();  // (N×D)

    // Step 2: sum exponentials across each row
    Eigen::VectorXd denominator = numerator.rowwise().sum();   // (N×1)

    // Step 3: divide each element in the row by that row’s sum
    Eigen::MatrixXd result = numerator.array().colwise() / denominator.array();


    return result ;
}
