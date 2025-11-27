// In the name of Allah
#pragma once 


#include <Eigen/Dense>
#include <iostream>

class CrossEntropyError {
    public:
    double calculate_error(Eigen::MatrixXd &prediction, Eigen::MatrixXd &actual) ; 

    private:
    Eigen::VectorXd select_correct_probs(const Eigen::MatrixXd& prediction,const Eigen::MatrixXd& actual) ; 


} ;