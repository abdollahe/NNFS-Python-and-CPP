// In the name of Allah

#include <Dense_Layer.h>

DenseLayer::DenseLayer(int num_of_neurons , int num_of_inputs) {
    this->weights.resize(num_of_inputs , num_of_neurons) ;
    this->weights.setRandom() ;
    this->bias.resize(1, num_of_neurons) ; 
    this->bias.setZero() ;
}

Eigen::MatrixXd DenseLayer::forward_pass(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd temp =  input * this->weights ;
    auto add_temp = temp.rowwise() + this->bias ;
    return add_temp ; 
}
