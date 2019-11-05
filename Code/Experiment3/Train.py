import numpy as np
import Utils as utils

def Train(data_preparer, neural_network, debugInfo):
    learning_rate = 0.1
    weight_distribution = 'Normal'
    epochs = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20])

    neural_network.InitWeights(weight_distribution)
    neural_network.SetLearningRate(learning_rate)

    for epoch in range(1, epochs.max() + 1):
        if debugInfo:
            print("Epoch " + str(epoch))

        utils.DoTrainRun(data_preparer, neural_network, debugInfo)

        if np.isin(epoch, epochs):
            utils.SaveObject(neural_network, "./Experiment3/Data/NeuralNetwork_" + str(epoch) + ".pkl")
