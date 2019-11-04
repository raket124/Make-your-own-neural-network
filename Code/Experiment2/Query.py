import numpy as np
import Utils as utils

def Query(data_preparer, debugInfo):
	record_count = data_preparer.GetCount()

	learning_rates = [0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.5, 0.7, 1]
	learning_rate_count = len(learning_rates);
	scores = np.zeros(learning_rate_count)

	print("Learning rate : Score")
	for lr in range(learning_rate_count):
		learning_rate = learning_rates[lr]
		neural_network = utils.LoadObject("./Experiment2/Data/NeuralNetwork_" + str(learning_rate) + ".pkl")

		score = 0
		for x in range(record_count):
			input = data_preparer.PrepareInput(x)
			output = data_preparer.PrepareOutput(x)
			result = neural_network.Query(input)

			expected_label = data_preparer.PrepareOutputLabel(x)
			result_label = np.argmax(result)
			if(expected_label == result_label):
				score += 1

		scores[lr] = score / record_count
		if debugInfo:
			print(str(learning_rate) + " : " + str(scores[lr]))

	utils.SaveObject(scores, "./Experiment2/Data/Scores.pkl")
