import sys

sys.path.append("./ABAGAIL.jar")

from java.util.concurrent import Executors, Callable, TimeUnit
from itertools import product
import time

import threading
import json

import base

import flipflop
import continuouspeaks
import tsp

import NNBackprop as NNbackprop
import NNGA as NNga
import NNRHC as NNrhc
import NNSA as NNsa

# MAX_CONCURRENT = 70

def shutdown_and_await_termination(pool, timeout):
    pool.shutdown()
    try:
        if not pool.awaitTermination(timeout, TimeUnit.SECONDS):
            pool.shutdownNow()
            if (not pool.awaitTermination(timeout, TimeUnit.SECONDS)):
                print >> sys.stderr, "Pool did not terminate"
    except InterruptedException, ex:
        # (Re-)Cancel if current thread also interrupted
        pool.shutdownNow()
        # Preserve interrupt status
        Thread.currentThread().interrupt()

pool = Executors.newWorkStealingPool()

TEST_CULLED = 'data/{}_test.csv'.format('Culled')
TRAIN_CULLED = 'data/{}_train.csv'.format('Culled')
VALIDATE_CULLED = 'data/{}_validate.csv'.format('Culled')

TEST_VEHICLE = 'data/{}_test.csv'.format('Vehicle')
TRAIN_VEHICLE = 'data/{}_train.csv'.format('Vehicle')
VALIDATE_VEHICLE = 'data/{}_validate.csv'.format('Vehicle')

experiment_data = [
    #([INPUT_LAYER, HIDDEN_LAYER1, ..., OUTPUT_LAYER], max iterations, test, train, validate, name)
    # OUTPUT should be 1 otherwise you get an index out of bounds error
    ([32, 32, 32, 1], 5001, TEST_CULLED, TRAIN_CULLED, VALIDATE_CULLED, 'Culled'),
    ([18, 18, 1], 5001, TEST_VEHICLE, TRAIN_VEHICLE, VALIDATE_VEHICLE, 'Vehicle')
    ]
rhc_args = [data for data in experiment_data]
print "RHC {}".format(len(rhc_args))
sa_args = [(CE, data[0], data[1], data[2], data[3], data[4], data[5]) for CE in [0.15, 0.35, 0.55, 0.70, 0.95] for data in experiment_data]
print "SA {}".format(len(sa_args))
ga_args = [(p, mate, mutate, data[0], data[1], data[2], data[3], data[4], data[5]) for p in [50] for mate in [20, 10] for mutate in [20, 10] for data in experiment_data]
print "GA {}".format(len(ga_args))
backprop_args = [data for data in experiment_data]
print "Backprop {}".format(len(backprop_args))

'''
for t in range(numTrials):
    for samples, keep, m in product([100], [50], [0.1, 0.3, 0.5, 0.7, 0.9]):
        run_mimic(t, samples, keep, m)
for t in range(numTrials):
    run_rhc(t)
for t in range(numTrials):
    for CE in [0.15, 0.35, 0.55, 0.75, 0.95]:
        run_sa(t, CE)
for t in range(numTrials):
    for pop, mate, mutate in product([100], [50, 30, 10], [50, 30, 10]):
        run_ga(t, pop, mate, mutate)
'''

numTrials = 5
toy_experiments = ['continuouspeaks', 'flipflop', 'tsp']
# toy_experiments = ['tsp']

toy_mimic = [{"type": mimic_type, "kind": "mimic", "args": (t, samples, keep, m)} for mimic_type in toy_experiments for t in range(numTrials) for samples, keep, m in product([100], [50], [0.1, 0.3, 0.5, 0.7, 0.9])]
print "Mimic {}".format(len(toy_mimic))
toy_rhc = [{"type": rhc_type, "kind": "rhc", "args": (t)} for rhc_type in toy_experiments for t in range(numTrials)]
print "RHC {}".format(len(toy_rhc))
toy_sa = [{"type": sa_type, "kind": "sa", "args": (t, CE)} for sa_type in toy_experiments for t in range(numTrials) for CE in [0.15, 0.35, 0.55, 0.75, 0.95]]
print "SA {}".format(len(toy_sa))
toy_ga = [{"type": ga_type, "kind": "ga", "args": (t, pop, mate, mutate)} for ga_type in toy_experiments for t in range(numTrials) for pop, mate, mutate in product([100], [50, 30, 10], [50, 30, 10])]
print "GA {}".format(len(toy_ga))

toys = toy_rhc + toy_sa + toy_ga + toy_mimic

TIMING_FILE = './output/timing.csv'

class RunExperiment(Callable):
    def __init__(self, experiment):
        self.experiment = experiment
        self.started = None
        self.completed = None
        self.result = None
        self.thread_used = None
        self.exception = None

    def __str__(self):
        if self.exception:
             return "[%s] %s download error %s in %.2fs" % \
                (self.thread_used, self.experiment, self.exception,
                 self.completed - self.started, ) #, self.result)
        elif self.completed:
            return "[%s] %s downloaded %dK in %.2fs" % \
                (self.thread_used, self.experiment, len(self.result)/1024,
                 self.completed - self.started, ) #, self.result)
        elif self.started:
            return "[%s] %s started at %s" % \
                (self.thread_used, self.experiment, self.started)
        else:
            return "[%s] %s not yet scheduled" % \
                (self.thread_used, self.experiment)

    # needed to implement the Callable interface;
    # any exceptions will be wrapped as either ExecutionException
    # or InterruptedException
    def call(self):
        self.thread_used = threading.currentThread().getName()
        self.started = time.time()
        if self.experiment['kind'] == 'rhc':
            exp_args = self.experiment['args']
        else:
            exp_args = ",".join(map(str,self.experiment['args']))
        print "START: {},{},{}".format(self.experiment['type'], self.experiment['kind'], exp_args)
        try:
            if self.experiment['type'] == 'continuouspeaks':
                if self.experiment['kind'] == 'rhc':
                    self.result = continuouspeaks.run_rhc(self.experiment['args'])
                if self.experiment['kind'] == 'sa':
                    self.result = continuouspeaks.run_sa(*self.experiment['args'])
                if self.experiment['kind'] == 'ga':
                    self.result = continuouspeaks.run_ga(*self.experiment['args'])
                if self.experiment['kind'] == 'mimic':
                    self.result = continuouspeaks.run_mimic(*self.experiment['args'])
            if self.experiment['type'] == 'flipflop':
                if self.experiment['kind'] == 'rhc':
                    self.result = flipflop.run_rhc(self.experiment['args'])
                if self.experiment['kind'] == 'sa':
                    self.result = flipflop.run_sa(*self.experiment['args'])
                if self.experiment['kind'] == 'ga':
                    self.result = flipflop.run_ga(*self.experiment['args'])
                if self.experiment['kind'] == 'mimic':
                    self.result = flipflop.run_mimic(*self.experiment['args'])
            if self.experiment['type'] == 'tsp':
                if self.experiment['kind'] == 'rhc':
                    self.result = tsp.run_rhc(self.experiment['args'])
                if self.experiment['kind'] == 'sa':
                    self.result = tsp.run_sa(*self.experiment['args'])
                if self.experiment['kind'] == 'ga':
                    self.result = tsp.run_ga(*self.experiment['args'])
                if self.experiment['kind'] == 'mimic':
                    self.result = tsp.run_mimic(*self.experiment['args'])
            if self.experiment['type'] == 'nn':
                if self.experiment['kind'] == 'backprop':
                    self.result = NNbackprop.main(*self.experiment['args'])
                if self.experiment['kind'] == 'ga':
                    self.result = NNga.main(*self.experiment['args'])
                if self.experiment['kind'] == 'rhc':
                    self.result = NNrhc.main(*self.experiment['args'])
                if self.experiment['kind'] == 'sa':
                    self.result = NNsa.main(*self.experiment['args'])
        except Exception, ex:
            self.exception = ex
        self.completed = time.time()
        if self.experiment['kind'] == 'rhc':
            exp_args = self.experiment['args']
        else:
            exp_args = ",".join(map(str,self.experiment['args']))
        finished = "{},{},{},{}".format(self.experiment['type'], self.experiment['kind'], exp_args, self.completed - self.started)
        print "  END: {}".format(finished)
        base.write_to_file(TIMING_FILE,"{}\n".format(finished))

        return self


threads = [
    RunExperiment({
        "type": "nn",
        "kind": "ga",
        "args": args
    }) for args in ga_args
] + [
    RunExperiment({
        "type": "nn",
        "kind": "rhc",
        "args": args
    }) for args in rhc_args
] + [
    RunExperiment({
        "type": "nn",
        "kind": "sa",
        "args": args
    }) for args in sa_args
] + [
    RunExperiment({
        "type": "nn",
        "kind": "backprop",
        "args": args
    }) for args in backprop_args
]
# + [
#     RunExperiment({
#         "type": toy['type'],
#         "kind": toy['kind'],
#         "args": toy['args']
#     }) for toy in toys
# ]

thread_count = len(threads)

print "Threads initialized: {}".format(len(threads))

pool.invokeAll(threads)

shutdown_and_await_termination(pool, 5)