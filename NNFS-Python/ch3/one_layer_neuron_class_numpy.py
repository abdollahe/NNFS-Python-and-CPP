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


# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Dense_Layer(num_of_input=2, num_of_neurons=3)

dense1.forward_pass(input_matrix=X)

# Let's see output of the first few samples:
print(dense1.output[:5])

