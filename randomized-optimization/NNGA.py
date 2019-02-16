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
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
from func.nn.backprop import RPROPUpdateRule
from opt.example import NeuralNetworkOptimizationProblem
from shared import SumOfSquaresError, DataSet, Instance
from func.nn.backprop import BackPropagationNetworkFactory
"""
GA NN training
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN3.py
import sys

sys.path.append("./ABAGAIL.jar")


# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

base.make_dirs(OUTPUT_DIRECTORY)

OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_LOG.csv'


def main(P, mate, mutate, layers, training_iterations, test_data_file, train_data_file, validate_data_file):
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
    oa_name = "GA_{}_{}_{}".format(P, mate, mutate)
    with open(OUTFILE.format(oa_name), 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst',
                                                            'elapsed'))
    classification_network = factory.createClassificationNetwork(
        layers, relu)
    nnop = NeuralNetworkOptimizationProblem(
        data_set, classification_network, measure)
    oa = StandardGeneticAlgorithm(P, mate, mutate, nnop)
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
    for p in [50]:
        for mate in [20, 10]:
            for mutate in [20, 10]:
                args = (p, mate, mutate, layers, training_iterations,
                        TEST_DATA_FILE, TRAIN_DATA_FILE, VALIDATE_DATA_FILE)
                main(*args)
