import numpy as np
import Utils as utils

def Train(data_preparer, neural_network, debugInfo):
    epochs = 13
    learning_rate = 0.025

    neural_network.InitWeights('Normal')
    neural_network.SetLearningRate(learning_rate)

    for epoch in range(1, epochs + 1):
        if debugInfo:
            print("Epoch " + str(epoch))

        utils.DoTrainRun(data_preparer, neural_network, debugInfo)

    utils.SaveObject(neural_network, "./Experiment5/Data/NeuralNetwork.pkl")
