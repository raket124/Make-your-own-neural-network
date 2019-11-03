import numpy as np
import Utils as utils

def Train(data_preparer, neural_network, debugInfo):
    learning_rates = np.array([0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.5, 0.7, 1])
    record_count = data_preparer.GetCount()

    for lr in learning_rates:
        neural_network.InitWeights('Normal')
        neural_network.SetLearningRate(lr)

        if debugInfo:
            print("Learning rate: " + str(lr))

        for x in range(record_count):
            input = data_preparer.PrepareInput(x)
            output = data_preparer.PrepareOutput(x)
            neural_network.Train(input, output)

            if debugInfo:
                utils.PrintDebugInfo(x, record_count)

        utils.SaveObject(neural_network, "./Experiment2/Data/NeuralNetwork_" + str(lr) + ".pkl")
