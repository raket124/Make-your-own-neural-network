import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Utils import LoadObject
from Utils import SaveObject

from NeuralNetwork import NeuralNetwork
from DataPreparer import DataPreparer

data_folder = "./Data/"
file_name = "mnist_test.csv"
data_preparer = DataPreparer(data_folder)

data_preparer.Read(file_name)
record_count = data_preparer.GetCount()

epochs = 15
learning_rates = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3])
learning_rates_count = len(learning_rates)

scores = np.zeros((epochs, learning_rates_count))

# print("Learning rate : Epochs : Score")
# for ep in range(epochs):
# 	epoch = ep + 1
# 	for lr in range(learning_rates_count):
# 		learning_rate = learning_rates[lr]
# 		neural_network = LoadObject("./Data/NeuralNetwork_4_" + str(learning_rate) + "_" + str(epoch) + ".pkl")
#
# 		score = 0
# 		for x in range(record_count):
# 			input = data_preparer.PrepareInput(x)
# 			output = data_preparer.PrepareOutput(x)
#
# 			result = neural_network.Query(input)
#
# 			expected_label = data_preparer.PrepareOutputLabel(x)
# 			result_label = np.argmax(result)
#
# 			if(expected_label == result_label):
# 				score += 1
#
# 		scores[ep, lr] = score / record_count
# 		print(str(learning_rate) + " : " + str(epoch) + " : " + str(scores[ep, lr]))
#
# print(scores)
# SaveObject(scores, "./Data/NeuralNetwork_4_scores.pkl")
scores = LoadObject("./Data/NeuralNetwork_4_scores.pkl")
print(scores)

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(learning_rates, range(1, epochs + 1))
ax.plot_surface(X, Y, scores)

ax.set_xlabel('Learning rate')
ax.set_ylabel('Epochs')
ax.set_zlabel('Score')

for angle in range(0, 181):
	ax.view_init(30, angle - 90)
	# plt.draw()
	plt.savefig('Output/Test/Surface' + str(angle) + '.png')
# xmax = epochs[scores.argmax()]
# ymax = scores.max()
#
# plt.plot(epochs, scores, 'Blue', zorder=1)
# dot = plt.scatter(xmax, ymax, s=None, c='Black', zorder=2)
# plt.title('Epochs experiment')
# plt.xlabel('Epochs')
# plt.ylabel('Score')
# plt.legend([dot], ['Best result\nEpoch: {:}\nScore: {:.2f}'.format(xmax, ymax)])
# plt.savefig('Output/Surface.png')
