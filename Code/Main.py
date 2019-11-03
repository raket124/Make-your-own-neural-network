import argparse
from Options import Options

from NeuralNetwork import NeuralNetwork
from DataPreparer import DataPreparer

from Experiment1.Train import Train as Train_1
from Experiment1.Query import Query as Query_1
from Experiment1.Result import Result as Result_1
from Experiment2.Train import Train as Train_2

parser = argparse.ArgumentParser()
parser.add_argument('Experiment', type=int, choices=range(1, 6))
parser.add_argument('Action', type=Options, choices=list(Options))
parser.add_argument('--DebugInfo', type=bool, default=True)
args = parser.parse_args()

actions = {
    (1, Options.Train) : Train_1,
    (1, Options.Query) : Query_1,
    (1, Options.Result) : Result_1,
    (2, Options.Train) : Train_2
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
