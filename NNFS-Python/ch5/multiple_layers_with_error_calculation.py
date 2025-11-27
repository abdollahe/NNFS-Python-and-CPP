#!/usr/bin/env python3
# In the name of Allah

import numpy as np

import nnfs
from nnfs.datasets import spiral_data

class Dense_Layer:
    def __init__(self, num_of_neurons : int, num_of_input : int):
        self.num_of_input = num_of_input
        self.num_of_neurons = num_of_neurons
        self.weight_matrix = np.random.randn(num_of_input,num_of_neurons) # We are creating the matrix transposed already!!!
        print(f"Shape of the weight matrix is: {self.weight_matrix.shape}")
        self.bias = np.zeros((1,num_of_neurons))


    def forward_pass(self, input_matrix):
        print(f"Shape of the input_matrix is: {input_matrix.shape}")
        mul_res = np.dot(input_matrix , self.weight_matrix)  
        print(f"The multiplication result is: {mul_res.size}") 
        self.output = mul_res + self.bias  
    

class Relu_Activation:
    def perform_activation(self, input_matrix):
        filtered = np.maximum(0,input_matrix)
        self.output = filtered
    

class Softmax_activation:
    def perform_activation(self, input_matrix):
        numerator = np.exp(input_matrix)
        denominator = np.sum(numerator, axis = 1, keepdims = True)
        self.output = np.divide(numerator , denominator)



class CrossEntropyLoss:
    def CalculateLoss(self, y_prediction, y_actual):
        neg_log = 0.0

        if len(y_actual.shape) == 1:
            neg_log = -np.log(np.clip(y_prediction[np.arange(len(y_prediction)), y_actual] , 1e-10, 1 - 1e-10))

        elif len(y_actual.shape) == 2:    
            neg_log = -np.log(np.clip(np.sum(y_prediction * y_actual , axis=1) , 1e-10 , 1 - 1e-10)) 

        
        average_loss = np.mean(neg_log)  

        print(f"average loss is: {average_loss}")

        return average_loss


# Create dataset
X, y = spiral_data(samples=1, classes=3)
print("X is:")
print(X)
print("======")
print("y is: ")
print(y)
# Create Dense layer with 2 input features and 3 output values
dense1 = Dense_Layer(num_of_input=2, num_of_neurons=3)
# Create ReLU activation (to be used with Dense layer):
activation1 = Relu_Activation()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Dense_Layer(num_of_input = 3, num_of_neurons = 3)
# Create Softmax activation (to be used with Dense layer):
softmax = Softmax_activation()
# Make a forward pass of our training data through this layer
dense1.forward_pass(X)
# Make a forward pass through activation function
# it takes the output of first dense layer here
activation1.perform_activation(dense1.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward_pass(activation1.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
softmax.perform_activation(dense2.output)

error = CrossEntropyLoss()

print(f"Final shape of output is: {softmax.output.shape}")
print(softmax.output)

res_loss = error.CalculateLoss(y_prediction = softmax.output , y_actual = y)

print(f"The loss is: {res_loss}")


