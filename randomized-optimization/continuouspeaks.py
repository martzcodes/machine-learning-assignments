import sys

sys.path.append("./ABAGAIL/ABAGAIL.jar")

import base
from itertools import product
from time import clock
from array import array
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.prob.MIMIC as MIMIC
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.DiscreteDependencyTree as DiscreteDependencyTree
import time


# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/continuouspeaks.py

"""
Commandline parameter(s):
   none
"""

OUTPUT_DIRECTORY = './output'
N = 100
T = 29
maxIters = 5001
numTrials = 5
fill = [2] * N
ranges = array('i', fill)

base.make_dirs(OUTPUT_DIRECTORY)

outfile = OUTPUT_DIRECTORY + '/CONTPEAKS/CONTPEAKS_{}_{}_LOG.csv'

# RHC

def run_rhc(t):
    fname = outfile.format('RHC', str(t + 1))
    ef = ContinuousPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 10)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        fevals = ef.fevals
        score = ef.value(rhc.getOptimal())
        ef.fevals -= 1
        st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
        # print fname, st
        base.write_to_file(fname,st)
    
    return

# SA


def run_sa(t, CE):
    fname = outfile.format('SA{}'.format(CE), str(t + 1))
    ef = ContinuousPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    sa = SimulatedAnnealing(1E10, CE, hcp)
    fit = FixedIterationTrainer(sa, 10)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        fevals = ef.fevals
        score = ef.value(sa.getOptimal())
        ef.fevals -= 1
        st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
        # print st
        base.write_to_file(fname,st)
    return

# GA


def run_ga(t, pop, mate, mutate):
    fname = outfile.format('GA{}_{}_{}'.format(
        pop, mate, mutate), str(t + 1))
    ef = ContinuousPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = SingleCrossOver()
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
    fit = FixedIterationTrainer(ga, 10)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        fevals = ef.fevals
        score = ef.value(ga.getOptimal())
        ef.fevals -= 1
        st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
        # print st
        base.write_to_file(fname,st)
    return

# MIMIC


def run_mimic(t, samples, keep, m):
    fname = outfile.format('MIMIC{}_{}_{}'.format(
        samples, keep, m), str(t + 1))
    ef = ContinuousPeaksEvaluationFunction(T)
    odd = DiscreteUniformDistribution(ranges)
    nf = DiscreteChangeOneNeighbor(ranges)
    mf = DiscreteChangeOneMutation(ranges)
    cf = SingleCrossOver()
    gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
    df = DiscreteDependencyTree(m, ranges)
    pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
    mimic = MIMIC(samples, keep, pop)
    fit = FixedIterationTrainer(mimic, 10)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        fevals = ef.fevals
        score = ef.value(mimic.getOptimal())
        ef.fevals -= 1
        st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
        # print st
        base.write_to_file(fname,st)
    return


if __name__ == '__main__':
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
