// In the name of Allah

#pragma once

#include <Eigen/Dense>

class ActivationFunctions {
    public:
        Eigen::MatrixXd relu_activation(const Eigen::MatrixXd &input) ;

        Eigen::MatrixXd softmax_activation(const Eigen::MatrixXd &input) ;
} ;
