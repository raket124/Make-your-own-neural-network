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
epochs = 15
learning_rates = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3])

neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
neural_network.InitActivation('Sigmoid')

data_preparer.Read(file_name)
count = data_preparer.GetCount()

for learning_rate in learning_rates:
    print("Learning rate " + str(learning_rate))
    neural_network.InitWeights('Normal')
    neural_network.SetLearningRate(learning_rate)

    for epoch in range(1, epochs + 1):
        print("Epoch " + str(epoch))
        for x in range(count):
        	input = data_preparer.PrepareInput(x)
        	output = data_preparer.PrepareOutput(x)

        	neural_network.Train(input, output)

        	if x == 0:
        		print(str(0) + "/" + str(count))
        	if (x + 1) % 1000 == 0:
        		print(str(x + 1) + "/" + str(count))

        SaveObject(neural_network, "./Data/NeuralNetwork_4_" + str(learning_rate) + "_" + str(epoch) + ".pkl")
