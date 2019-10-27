import numpy as np
import matplotlib.pyplot as plt
from Utils import SaveObject

from NeuralNetwork import NeuralNetwork
from DataPreparer import DataPreparer

data_folder = "./Data/"
file_name = "mnist_train.csv"
data_preparer = DataPreparer(data_folder)

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rates = [0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.5, 0.7, 1]

neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
neural_network.InitWeights('Normal')
neural_network.InitActivation('Sigmoid')

data_preparer.Read(file_name)
count = data_preparer.GetCount()

for lr in learning_rates:
    neural_network.SetLearningRate(lr)
    print("Learning rate " + str(lr))
    for x in range(count):
    	input = data_preparer.PrepareInput(x)
    	output = data_preparer.PrepareOutput(x)

    	neural_network.Train(input, output)

    	if x == 0:
    		print(str(0) + "/" + str(count))
    	if (x + 1) % 1000 == 0:
    		print(str(x + 1) + "/" + str(count))

    SaveObject(neural_network, "./Data/NeuralNetwork_2_" + str(lr) + ".pkl")