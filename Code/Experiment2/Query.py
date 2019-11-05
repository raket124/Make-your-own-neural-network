import numpy as np
import Utils as utils

def Query(data_preparer, debugInfo):
	learning_rates = [0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.5, 0.7, 1]
	learning_rate_count = len(learning_rates);
	scores = np.zeros(learning_rate_count)

	if debugInfo:
		print("Learning rate : Score")
	for lr in range(learning_rate_count):
		learning_rate = learning_rates[lr]
		neural_network = utils.LoadObject("./Experiment2/Data/NeuralNetwork_" + str(learning_rate) + ".pkl")

		scores[lr] = utils.DoQueryRun(data_preparer, neural_network, debugInfo)

		if debugInfo:
			print(str(learning_rate) + " : " + str(scores[lr]))

	utils.SaveObject(scores, "./Experiment2/Data/Scores.pkl")
