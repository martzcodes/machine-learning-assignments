import sys

sys.path.append("./ABAGAIL.jar")

import base
from java.lang import Math
from shared import Instance
import random as rand
import time
import os
import csv
from func.nn.activation import RELU
import opt.SimulatedAnnealing as SimulatedAnnealing
from func.nn.backprop import RPROPUpdateRule
from opt.example import NeuralNetworkOptimizationProblem
from shared import SumOfSquaresError, DataSet, Instance
from func.nn.backprop import BackPropagationNetworkFactory
"""
SA NN training on HTRU2 data
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN2.py


# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

base.make_dirs(OUTPUT_DIRECTORY)

# Network parameters found "optimal" in Assignment 1
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_LOG.csv'


def main(CE, layers, training_iterations, test_data_file, train_data_file, validate_data_file):
    """Run this experiment"""
    training_ints = base.initialize_instances(train_data_file)
    testing_ints = base.initialize_instances(test_data_file)
    validation_ints = base.initialize_instances(validate_data_file)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()
    # 50 and 0.000001 are the defaults from RPROPUpdateRule.java
    rule = RPROPUpdateRule(0.064, 50, 0.000001)
    oa_name = "SA_{}".format(CE)
    with open(OUTFILE.format(oa_name), 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst',
                                                            'elapsed'))
    classification_network = factory.createClassificationNetwork(
        layers, relu)
    nnop = NeuralNetworkOptimizationProblem(
        data_set, classification_network, measure)
    oa = SimulatedAnnealing(1E10, CE, nnop)
    base.train(oa, classification_network, oa_name, training_ints, validation_ints, testing_ints, measure,
               training_iterations, OUTFILE.format(oa_name))
    return


if __name__ == "__main__":
    DS_NAME = 'Culled'
    TEST_DATA_FILE = 'data/{}_test.csv'.format(DS_NAME)
    TRAIN_DATA_FILE = 'data/{}_train.csv'.format(DS_NAME)
    VALIDATE_DATA_FILE = 'data/{}_validate.csv'.format(DS_NAME)
    layers = [250, 250, 1]
    training_iterations = 5001
    for CE in [0.15, 0.35, 0.55, 0.70, 0.95]:
        main(CE, layers, training_iterations, TEST_DATA_FILE,
             TRAIN_DATA_FILE, VALIDATE_DATA_FILE)
