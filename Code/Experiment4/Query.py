import numpy as np
import Utils as utils

def Query(data_preparer, debugInfo):
	epochs = 15
	learning_rates = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3])
	learning_rates_count = len(learning_rates)
	scores = np.zeros((epochs, learning_rates_count))

	if debugInfo:
		print("Learning rate : Epochs : Score")
	for ep in range(epochs):
		epoch = ep + 1
		for lr in range(learning_rates_count):
			learning_rate = learning_rates[lr]

			neural_network = utils.LoadObject("./Experiment4/Data/NeuralNetwork_" + str(learning_rate) + "_" + str(epoch) + ".pkl")

			scores[ep, lr] = utils.DoQueryRun(data_preparer, neural_network, debugInfo)

			if debugInfo:
				print(str(learning_rate) + " : " + str(epoch) + " : " + str(scores[ep, lr]))

	utils.SaveObject(scores, "./Experiment4/Data/Scores.pkl")
