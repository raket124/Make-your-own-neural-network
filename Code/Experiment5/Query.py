import numpy as np
import Utils as utils
import matplotlib.pyplot as plt

def Query(data_preparer, debugInfo):
    neural_network = utils.LoadObject("./Experiment5/Data/NeuralNetwork.pkl")

    result = []
    for x in range(10):
        output_label = data_preparer.PrepareOutput(x);
        estimated_input = neural_network.ReverseQuery(output_label);
        result.append(estimated_input)
    results = np.array(result).reshape((10, 784))
    
    utils.SaveObject(results, "./Experiment5/Data/Results.pkl")
