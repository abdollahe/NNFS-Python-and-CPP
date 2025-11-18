// In the name of Allah

#include <Activation_Functions.h>
#include <helper.h>
#include <Dense_Layer.h>
#include <iostream>


int main() {
   
    // Create dataset
    auto data = create_data(100 , 3) ;

    // Create Dense layer with 2 input features and 3 output values
    DenseLayer* layer1 = new DenseLayer(3 ,2) ;
  
    // Create activation object
    ActivationFunctions* activations = new ActivationFunctions() ;
    
    // Make a forward pass of our training data through first layer
    auto output1 = layer1->forward_pass(data.first) ;

    // Make a forward pass through activation function
    //it takes the output of first dense layer here
    auto output1_activated = activations->relu_activation(output1) ;


    std :: cout << "The shape of the final output is:" << output1_activated.rows() << "x" << output1_activated.cols() << std::endl ;
    
    // std:: cout << output1_activated << std::endl; 

    return 0 ;

}