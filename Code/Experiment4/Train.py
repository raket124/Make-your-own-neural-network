import numpy as np
import Utils as utils

def Train(data_preparer, neural_network, debugInfo):
    epochs = 15
    learning_rates = np.array([0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3])

    for learning_rate in learning_rates:
        if debugInfo:
            print("Learning rate: " + str(learning_rate))

        neural_network.InitWeights('Normal')
        neural_network.SetLearningRate(learning_rate)

        for epoch in range(1, epochs + 1):
            if debugInfo:
                print("Epoch " + str(epoch))

            utils.DoTrainRun(data_preparer, neural_network, debugInfo)

            utils.SaveObject(neural_network, "./Experiment4/Data/NeuralNetwork_" + str(learning_rate) + "_" + str(epoch) + ".pkl")
