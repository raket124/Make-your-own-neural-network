import numpy as np
import matplotlib.pyplot as plt
from Utils import LoadObject

from NeuralNetwork import NeuralNetwork
from DataPreparer import DataPreparer

data_folder = "./Data/"
file_name = "mnist_test.csv"
data_preparer = DataPreparer(data_folder)

data_preparer.Read(file_name)
record_count = data_preparer.GetCount()

epochs = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20])
epochs_count = len(epochs);
scores = np.zeros(epochs_count)

print("Epochs : Score")
for ep in range(epochs_count):
	epoch = epochs[ep]
	neural_network = LoadObject("./Data/NeuralNetwork_3_" + str(epoch) + ".pkl")
	score = 0

	for x in range(record_count):
		input = data_preparer.PrepareInput(x)
		output = data_preparer.PrepareOutput(x)

		result = neural_network.Query(input)

		expected_label = data_preparer.PrepareOutputLabel(x)
		result_label = np.argmax(result)

		if(expected_label == result_label):
			score += 1

	scores[ep] = score / record_count
	print(str(epoch) + " : " + str(scores[ep]))

xmax = epochs[scores.argmax()]
ymax = scores.max()

plt.plot(epochs, scores, 'Blue', zorder=1)
dot = plt.scatter(xmax, ymax, s=None, c='Black', zorder=2)
plt.title('Epochs experiment')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend([dot], ['Best result\nEpoch: {:}\nScore: {:.2f}'.format(xmax, ymax)])
plt.savefig('Output/Epoch.png')
