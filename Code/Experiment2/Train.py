import numpy as np
import Utils as utils

def Train(data_preparer, neural_network, debugInfo):
    learning_rates = np.array([0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.5, 0.7, 1])
    weight_distribution = 'Normal'

    for lr in learning_rates:
        if debugInfo:
            print("Learning rate: " + str(lr))

        neural_network.InitWeights(weight_distribution)
        neural_network.SetLearningRate(lr)
        utils.DoTrainRun(data_preparer, neural_network, debugInfo)

    utils.SaveObject(neural_network, "./Experiment2/Data/NeuralNetwork_" + str(lr) + ".pkl")
