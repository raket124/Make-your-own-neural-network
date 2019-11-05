import numpy as np
import Utils as utils

def Query(data_preparer, debugInfo):
	epochs = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20])
	epochs_count = len(epochs);
	scores = np.zeros(epochs_count)

	if debugInfo:
		print("Epochs : Score")
	for ep in range(epochs_count):
		epoch = epochs[ep]
		neural_network = utils.LoadObject("./Experiment3/Data/NeuralNetwork_" + str(epoch) + ".pkl")

		scores[ep] = utils.DoQueryRun(data_preparer, neural_network, debugInfo)

		if debugInfo:
			print(str(epoch) + " : " + str(scores[ep]))

	utils.SaveObject(scores, "./Experiment3/Data/Scores.pkl")
