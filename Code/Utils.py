import dill

def SaveObject(obj, filename):
    with open(filename, 'wb') as output:
        dill.dump(obj, output)

def LoadObject(filename):
    with open(filename, 'rb') as output:
        return dill.load(output)
