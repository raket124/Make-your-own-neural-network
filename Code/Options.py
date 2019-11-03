from enum import Enum

class Options(Enum):
    Train = "Train"
    Query = "Query"
    Result = "Result"

    def __str__(self):
        return self.value
