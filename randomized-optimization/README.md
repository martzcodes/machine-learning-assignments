# Assignment 2 - Randomized Optimization

## _USE AT YOUR OWN RISK_

(I haven't reviewed the output yet)

## General

This is heavily modified from cmaron's code https://github.com/cmaron/CS-7641-assignments

I added in threading so I could run it on AWS....

The code for this assignment chooses three toy problems, but there are other options available in _ABAGAIL_.

If you are running this code in OS X you should consider downloading Jython directly. The version provided by homebrew does mot seem to work as expected for this code.

## Usage

Because _ABAGAIL_ does not implement cross validation some work must be done on the dataset before the other code can be run. Setup your loaders in run_experiment.py and generate the data with

```
python run_experiment.py --dump_data
```

Configure your params from A1 in `run_jython.py` and then run:

Increase the default memory: `alias jythonMem="java -Xmx4096m -Xss1024m -classpath /home/ubuntu/jython/jython.jar: -Dpython.home/home/ubuntu/jython -Dpython.executable=/home/ubuntu/jython/bin/jython org.python.util.jython"` (adjust as necessary)

```
jythonMem run_jython.py
```

^^ Heavy emphasis on the _jython_ part of that. This will (supposedly) fully utilize your available threads. I ran it on an AWS c5.9xlarge which costs \$1.53/hour at the time of this writing. Adjust memory as needed

## Data

The data loading code expects datasets to be stored in "./data"... although this is configurable in the run_jython.py file.

Because _ABAGAIL_ does not implement cross validation some work must be done on the dataset before the other code can
be run. The data can be generated via

```
python run_experiment.py --dump_data
```

Be sure to run this before running any of the experiments.

## Output

Output CSVs and images are written to `./output` and `./output/images` respectively. Sub-folders will be created for
each toy problem (`CONTPEAKS`, `FLIPFLOP`, `TSP`) and the neural network from the _Supervised Learning Project_ (`NN_OUTPUT`, `NN`).

If these folders do not exist the experiments module will attempt to create them.

## Running Experiments

Each experiment can be run as a separate script. Running the actual optimization algorithms to generate data requires
the use of Jython.

For the three toy problems, run:

- continuoutpeaks.py
- flipflop.py
- tsp.py

For the neural network problem, run:

- NN-Backprop.py
- NN-GA.py
- NN-RHC.py
- NN-SA.py

## Graphing

The `plotting.py` script takes care of all the plotting. Since the files output from the scripts above follow a common
naming scheme it will determine the problem, algorithm, and parameters as needed and write the output to sub-folders in
`./output/images`. This _must_ be run via python, specifically an install of python that has the requirements from
`requirements.txt` installed.

In addition to the images, a csv file of the best parameters per problem/algorithm pair is written to
`./output/best_results.csv`
