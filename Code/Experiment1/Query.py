import numpy as np
import Utils as utils

def Query(data_preparer, debugInfo):
	neural_network = utils.LoadObject("./Experiment1/Data/NeuralNetwork.pkl")
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

		if debugInfo:
			utils.PrintDebugInfo(x, record_count)

	scores = np.array([score / record_count])
	utils.SaveObject(score, "./Experiment1/Data/Scores.pkl")
