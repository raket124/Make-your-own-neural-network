import numpy as np
import Utils as utils

def Query(data_preparer, debugInfo):
	neural_network = utils.LoadObject("./Experiment1/Data/NeuralNetwork.pkl")

	score = utils.DoQueryRun(data_preparer, neural_network, debugInfo)

	utils.SaveObject(score, "./Experiment1/Data/Scores.pkl")
