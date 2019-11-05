import numpy as np
import dill

def SaveObject(obj, filename):
    with open(filename, 'wb') as output:
        dill.dump(obj, output)

def LoadObject(filename):
    with open(filename, 'rb') as output:
        return dill.load(output)

def PrintDebugInfo(current_index, max_index, steps = 10):
    step = max_index / steps
    if current_index == 0:
        print(str(current_index) + "/" + str(max_index))
    if (current_index + 1) % step == 0:
        print(str(current_index + 1) + "/" + str(max_index))

def DoTrainRun(data_preparer, neural_network, debugInfo):
    record_count = data_preparer.GetCount()
    for x in range(record_count):
        input = data_preparer.PrepareInput(x)
        output = data_preparer.PrepareOutput(x)
        neural_network.Train(input, output)
        if debugInfo:
            PrintDebugInfo(x, record_count)

def DoQueryRun(data_preparer, neural_network, debugInfo):
    score = 0
    record_count = data_preparer.GetCount()
    for x in range(record_count):
        input = data_preparer.PrepareInput(x)
        output = data_preparer.PrepareOutput(x)
        result = neural_network.Query(input)

        expected_label = data_preparer.PrepareOutputLabel(x)
        result_label = np.argmax(result)
        if(expected_label == result_label):
            score += 1
    return float(score / record_count)
