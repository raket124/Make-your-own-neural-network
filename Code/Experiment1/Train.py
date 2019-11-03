import numpy as np
import Utils as utils

def Train(data_preparer, neural_network, debugInfo):
    learning_rate = 0.3
    weight_distribution = 'Normal'

    neural_network.SetLearningRate(learning_rate)
    neural_network.InitWeights(weight_distribution)
    record_count = data_preparer.GetCount()

    for x in range(record_count):
        input = data_preparer.PrepareInput(x)
        output = data_preparer.PrepareOutput(x)
        neural_network.Train(input, output)

        if debugInfo:
            utils.PrintDebugInfo(x, record_count)

    utils.SaveObject(neural_network, "./Experiment1/Data/NeuralNetwork.pkl")
