import argparse
from Options import Options

from NeuralNetwork import NeuralNetwork
from DataPreparer import DataPreparer

from Experiment1.Train import Train as Train_1
from Experiment1.Query import Query as Query_1
from Experiment1.Result import Result as Result_1
from Experiment2.Train import Train as Train_2
from Experiment2.Query import Query as Query_2
from Experiment2.Result import Result as Result_2
from Experiment3.Train import Train as Train_3
from Experiment3.Query import Query as Query_3
from Experiment3.Result import Result as Result_3
from Experiment4.Train import Train as Train_4
from Experiment4.Query import Query as Query_4
from Experiment4.Result import Result as Result_4
from Experiment5.Train import Train as Train_5
from Experiment5.Query import Query as Query_5
from Experiment5.Result import Result as Result_5

parser = argparse.ArgumentParser()
parser.add_argument('Experiment', type=int, choices=range(1, 6))
parser.add_argument('Action', type=Options, choices=list(Options))
parser.add_argument('--DebugInfo', type=bool, default=True)
args = parser.parse_args()

actions = {
    (1, Options.Train) : Train_1,
    (1, Options.Query) : Query_1,
    (1, Options.Result) : Result_1,
    (2, Options.Train) : Train_2,
    (2, Options.Query) : Query_2,
    (2, Options.Result) : Result_2,
    (3, Options.Train) : Train_3,
    (3, Options.Query) : Query_3,
    (3, Options.Result) : Result_3,
    (4, Options.Train) : Train_4,
    (4, Options.Query) : Query_4,
    (4, Options.Result) : Result_4,
    (5, Options.Train) : Train_5,
    (5, Options.Query) : Query_5,
    (5, Options.Result) : Result_5,
}

data_folder = "./Data/"
data_train = "mnist_train.csv"
data_query = "mnist_test.csv"

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
activation_function = 'Sigmoid'

neural_network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
neural_network.InitActivation(activation_function)

data_preparer = DataPreparer(data_folder)

if args.Action == Options.Train:
    data_preparer.Read(data_train)
    actions[(args.Experiment, args.Action)](data_preparer, neural_network, args.DebugInfo)
if args.Action == Options.Query:
    data_preparer.Read(data_query)
    actions[(args.Experiment, args.Action)](data_preparer, args.DebugInfo)
if args.Action == Options.Result:
    actions[(args.Experiment, args.Action)]()
