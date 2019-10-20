import numpy as np
import matplotlib.pyplot as plt
from Utils import LoadObject

from NeuralNetwork import NeuralNetwork
from DataPreparer import DataPreparer

data_folder = "./Data/"
file_name = "mnist_test.csv"
data_preparer = DataPreparer(data_folder)

neural_network = LoadObject("./Data/NeuralNetwork.pkl")

data_preparer.Read(file_name)
count = data_preparer.GetCount()

score = np.zeros(count)

for x in range(count):
	input = data_preparer.PrepareInput(x)
	output = data_preparer.PrepareOutput(x)

	result = neural_network.Query(input)

	expected_label = data_preparer.PrepareOutputLabel(x)
	result_label = np.argmax(result)

	if(expected_label == result_label):
		score[x] = 1

print(score.sum() / count)

# plt.imsave(str(x) + '.png', img, cmap='gray')
