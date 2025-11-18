// In the name of Allah

#include <Activation_Functions.h>
#include <helper.h>
#include <Dense_Layer.h>
#include <iostream>


int main() {
   
    // Create dataset
    auto data = create_data(1 , 3) ;

    // Create Dense layer with 2 input features and 3 output values
    DenseLayer* layer1 = new DenseLayer(3 ,2) ;

    // Create second Dense layer with 3 input features (as we take output 
    // of previous layer here) and 3 output values
    DenseLayer* layer2 = new DenseLayer(3,3) ;
    
    // Create activation object
    ActivationFunctions* activations = new ActivationFunctions() ;
    
    // Make a forward pass of our training data through first layer
    auto output1 = layer1->forward_pass(data.first) ;

    // Make a forward pass through activation function
    //it takes the output of first dense layer here
    auto output1_activated = activations->relu_activation(output1) ;

    // Make a forward pass through second Dense layer
    // it takes outputs of activation function of first layer as inputs
    auto output2 = layer2->forward_pass(output1_activated) ;

    // Make a forward pass through activation function
    // it takes the output of second dense layer here
    auto output2_activated = activations->softmax_activation(output2) ;

    std :: cout << "The shape of the final output is:" << output2_activated.rows() << "x" << output2_activated.cols() << std::endl ;
    
    std:: cout << output2_activated << std::endl; 

    return 0 ;

}