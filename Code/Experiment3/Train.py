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
learning_rate = 0.1
epochs = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20])

neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
neural_network.InitWeights('Normal')
neural_network.InitActivation('Sigmoid')
neural_network.SetLearningRate(learning_rate)

data_preparer.Read(file_name)
count = data_preparer.GetCount()

for epoch in range(1, epochs.max() + 1):
    print("Epoch " + str(epoch))
    for x in range(count):
    	input = data_preparer.PrepareInput(x)
    	output = data_preparer.PrepareOutput(x)

    	neural_network.Train(input, output)

    	if x == 0:
    		print(str(0) + "/" + str(count))
    	if (x + 1) % 1000 == 0:
    		print(str(x + 1) + "/" + str(count))

    if np.isin(epoch, epochs):
        SaveObject(neural_network, "./Data/NeuralNetwork_3_" + str(epoch) + ".pkl")
