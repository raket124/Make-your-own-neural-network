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
