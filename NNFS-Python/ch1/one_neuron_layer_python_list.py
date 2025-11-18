#!/usr/bin/env python3
# In the name of Allah
'''
This is a simple Python script to create a simple neuron using the basid Python lists
'''


# One neuron definition using lists

inputs = [1, 2, 3, 4]
weights = [0.4, 0.3, 0.2, 0.1]
bias = 2

neuron = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias

print(neuron)


# One layer consisting of 4 neurons

weights1 = [0.2, 0.3, 0.4, 0.1]
bias1 = 2
neuron1 = inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1

weights2 = [0.3, 0.2, 0.1, 0.4]
bias2 = 2
neuron2 = inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2

weights3 = [0.6, 0.7, 0.8, 0.9]
bias3 = 2
neuron3 = inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3

weights4 = [0.9, 0.8, 0.7, 0.6]
bias4 = 2
neuron4 = inputs[0]*weights4[0] + inputs[1]*weights4[1] + inputs[2]*weights4[2] + inputs[3]*weights4[3] + bias4

outputs = [neuron1,neuron2,neuron3,neuron4]

print(outputs)



