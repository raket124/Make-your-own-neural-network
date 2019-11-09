import numpy as np
import scipy.special

class NeuralNetwork:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = output_nodes

	def SetLearningRate(self, learning_rate):
		self.learning_rate = learning_rate

	def InitWeights(self, algorithm):
		if algorithm == 'Rand':
			self.weights_input_hidden = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
			self.weights_hidden_output = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
		if algorithm == 'Normal':
			self.weights_input_hidden = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
			self.weights_hidden_output = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

	def InitActivation(self, algorithm):
		if algorithm == 'Sigmoid':
			self.activation_function = lambda x: scipy.special.expit(x)
			self.inverse_activation_function = lambda x: scipy.special.logit(x)

	def Train(self, inputs_list, targets_list):
		inputs = np.array(inputs_list, ndmin=2).T
		targets = np.array(targets_list, ndmin=2).T

		hidden_inputs = np.dot(self.weights_input_hidden, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

		self.weights_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
		self.weights_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

	def Query(self, inputs_list):
		inputs = np.array(inputs_list, ndmin=2).T

		hidden_inputs = np.dot(self.weights_input_hidden, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

	def ReverseQuery(self, targets_list):
		final_outputs = np.array(targets_list, ndmin=2).T

		final_inputs = self.inverse_activation_function(final_outputs)
		hidden_outputs = np.dot(self.weights_hidden_output.T, final_inputs)

		hidden_outputs -= np.min(hidden_outputs)
		hidden_outputs /= np.max(hidden_outputs)
		hidden_outputs *= 0.98
		hidden_outputs += 0.01

		hidden_inputs = self.inverse_activation_function(hidden_outputs)
		inputs = np.dot(self.weights_input_hidden.T, hidden_inputs)

		inputs -= np.min(inputs)
		inputs /= np.max(inputs)
		inputs *= 0.98
		inputs += 0.01

		return inputs
