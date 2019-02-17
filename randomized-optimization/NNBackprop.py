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
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
from shared import SumOfSquaresError, DataSet, Instance
from func.nn.backprop import BackPropagationNetworkFactory
"""
Backprop NN training
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN0.py

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'

base.make_dirs(OUTPUT_DIRECTORY)

# Network parameters found "optimal" in Assignment 1
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_{}_LOG.csv'


def main(layers, training_iterations, test_data_file, train_data_file, validate_data_file, data_name):
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
    oa_names = ["Backprop"]
    classification_network = factory.createClassificationNetwork(layers, relu)
    base.train(BatchBackPropagationTrainer(data_set, classification_network, measure, rule), classification_network,
               'Backprop', training_ints, validation_ints, testing_ints, measure, training_iterations,
               OUTFILE.format(data_name, 'Backprop'))
    return


if __name__ == "__main__":
    DS_NAME = 'Vehicle'
    TEST_DATA_FILE = 'data/{}_test.csv'.format(DS_NAME)
    TRAIN_DATA_FILE = 'data/{}_train.csv'.format(DS_NAME)
    VALIDATE_DATA_FILE = 'data/{}_validate.csv'.format(DS_NAME)
    # [([32, 32, 32, 1], 5001, TEST_CULLED, TRAIN_CULLED, VALIDATE_CULLED, 'Culled'), ([18, 18, 4], 5001, TEST_VEHICLE, TRAIN_VEHICLE, VALIDATE_VEHICLE, 'Vehicle')]
    main([18, 18, 1], 5001, TEST_DATA_FILE,
         TRAIN_DATA_FILE, VALIDATE_DATA_FILE, DS_NAME)
