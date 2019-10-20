import numpy as np

class DataPreparer:
    def __init__(self, path):
        self.__path = path;
        self.__count = -1

    def Read(self, file):
        file_path = self.__path + file
        self.__lines = self.__GetLines(file_path)
        self.__count = len(self.__lines)

    def GetCount(self):
        return self.__count

    def PrepareOutputLabel(self, index):
        return int(self.__lines[index].split(',')[0])

    def PrepareOutput(self, index):
        number = self.PrepareOutputLabel(index)

        data = np.zeros(10) + 0.01
        data[number] = 0.99
        return data

    def PrepareInput(self, index):
        data = self.__lines[index].split(',')[1:]

        data = np.asfarray(data)
        data = (data / 255 * 0.99) + 0.01
        return data

    def __GetLines(self, file_path):
        file = open(file_path, 'r')
        lines = file.readlines()
        file.close()
        return lines
