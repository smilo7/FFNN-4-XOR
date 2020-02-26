#backprop xor net

import numpy as np

INPUTS = np.array([[0,0], [1,0], [0,1], [1,1]])
XOR_OUTPUTS = np.array([[0],[1],[1],[0]])


class Multilayer_Network:

	def __init__(self, inputs, expected_output,epochs=20000, l_rate=0.1):
		self.input = inputs
		self.expected_output = expected_output
		self.epochs = epochs
		self.l_rate = l_rate
		self.il_num = 2
		self.hl_num = 2
		self.ol_num = 1
		#il stands for input layer hl stands for hidden layer, ol stands for output layer
		self.hl_weights = np.random.uniform(size=(self.il_num, self.hl_num))
		self.hl_bias = np.random.uniform(size=(1, self.hl_num))
		self.ol_weights = np.random.uniform(size=(self.hl_num, self.ol_num))
		self.ol_bias = np.random.uniform(size=(1, self.ol_num))

	#sigmoid functions
	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def sigmoid_deriv(self, x):
		return x * (1-x)

	def train(self):
		for cycle in range(self.epochs):
			#propogating forward yee
			hl_activation = np.dot(self.input, self.hl_weights) 
			hl_activation += self.hl_bias
			hl_output = self.sigmoid(hl_activation)

			#then take the output for the hidden layer (previous to the output)
			#and use it in the output layer
			ol_activation = np.dot(hl_output, self.ol_weights) 
			ol_activation += self.ol_bias
			ol_predicted = self.sigmoid(ol_activation)

			#backpropogation
			ol_error = self.expected_output - ol_predicted
			ol_deriv = ol_error * self.sigmoid_deriv(ol_predicted)
			
			hl_error = ol_deriv.dot(self.ol_weights.T)
			hl_deriv = hl_error * self.sigmoid_deriv(hl_output)

			#update weights n bias
			self.ol_weights += hl_output.T.dot(ol_deriv) * self.l_rate
			self.ol_bias += np.sum(ol_deriv, axis=0, keepdims=True) * self.l_rate
			self.hl_weights += self.input.T.dot(hl_deriv) * self.l_rate
			self.hl_bias += np.sum(hl_deriv, axis=0, keepdims=True) * self.l_rate
			#print(ol_predicted)

	def predict(self, test_inputs):
		hl_activation = np.dot(test_inputs, self.hl_weights) 
		hl_activation += self.hl_bias
		hl_output = self.sigmoid(hl_activation)

		#then take the output for the hidden layer (previous to the output)
		#and use it in the output layer
		ol_activation = np.dot(hl_output, self.ol_weights) 
		ol_activation += self.ol_bias
		ol_predicted = self.sigmoid(ol_activation)
		print(ol_predicted)
		return np.rint(ol_predicted)



test = Multilayer_Network(INPUTS, XOR_OUTPUTS)
test.train()
print(test.predict(INPUTS))