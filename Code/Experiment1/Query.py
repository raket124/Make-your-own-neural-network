import numpy as np
import matplotlib.pyplot as plt
from Utils import LoadObject

from NeuralNetwork import NeuralNetwork
from DataPreparer import DataPreparer

data_folder = "./Data/"
file_name = "mnist_test.csv"
data_preparer = DataPreparer(data_folder)

neural_network = LoadObject("./Data/NeuralNetwork_1.pkl")

data_preparer.Read(file_name)
record_count = data_preparer.GetCount()

score = 0
for x in range(record_count):
	input = data_preparer.PrepareInput(x)
	output = data_preparer.PrepareOutput(x)

	result = neural_network.Query(input)

	expected_label = data_preparer.PrepareOutputLabel(x)
	result_label = np.argmax(result)

	if(expected_label == result_label):
		score += 1

print("Result " + str(score / record_count))
