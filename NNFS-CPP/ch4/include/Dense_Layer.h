// In the name of Allah
#pragma once 

#include <Eigen/Dense>


class DenseLayer {
    private:
        Eigen::MatrixXd weights ;
        Eigen::RowVectorXd bias ; 
    
    public:
        DenseLayer(int num_of_neurons , int num_of_inputs) ;
        
        Eigen::MatrixXd forward_pass(const Eigen::MatrixXd &input) ;
     
};

