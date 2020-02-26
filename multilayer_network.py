#backprop xor net

import numpy as np


'''sigmoid functions'''
def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_der(x):
	return x * (1-x)


INPUTS = np.array([[0,0], [1,0], [0,1], [1,1]])
XOR_OUTPUTS = [0,1,1,0]


class Multilayer_Network:

	def __init__(self, inputs, epochs, l_rate=0.1):
		self.input = inputs
		self.epochs = epochs
		self.l_rate = l_rate
		self.input_layer_num = 2
		self.hidden_layer_num = 2
		self.output_layer_num = 1
		self.hidden_weights = np.random.uniform(size=(self.input_layer_num, hidden_layer_num))
		self.hidden_bias = np.random.uniform(size=(1, self.output_layer_num))
	

	def train(self):


