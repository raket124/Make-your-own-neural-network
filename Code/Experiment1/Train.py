import numpy as np
import Utils as utils

def Train(data_preparer, neural_network, debugInfo):
    learning_rate = 0.3
    weight_distribution = 'Normal'

    neural_network.SetLearningRate(learning_rate)
    neural_network.InitWeights(weight_distribution)

    utils.DoTrainRun(data_preparer, neural_network, debugInfo)

    utils.SaveObject(neural_network, "./Experiment1/Data/NeuralNetwork.pkl")
