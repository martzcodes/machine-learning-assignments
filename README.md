# OMSCS Machine Learning Assignments

This repo is full of code for [CS 7641 - Machine Learning](https://www.omscs.gatech.edu/cs-7641-machine-learning) at Georgia Tech.

A huge thanks to Chad Maron ([https://github.com/cmaron](https://github.com/cmaron)) for sharing his code. Much of the code contained in this repo is based off of his work. I made some slight improvements but it's largely unmodified.

A huge thanks to Jonathan Tay ([https://github.com/JonathanTay](https://github.com/JonathanTay)) for sharing his code too... Chad's code was based on his.

## Wait, code?

Yup, we are encouraged to steal code. All the code. It's fine. Only the analysis matters.

For more support of this claim, see [https://gist.github.com/cmaron/46f0992d42be87380c208086eec9797f](https://gist.github.com/cmaron/46f0992d42be87380c208086eec9797f)

## How do I use this?

If a python virtual environment has been setup for the project, a simple `pip install -r requirements.txt` should take care of the required packages.

Each assignment folder has its own `run_experiment.py` that will do most of the work for you.

Running `python run_experiment.py -h` should provide a list of options for what you can do.

For the most part it is simple to run a given set of experiments based on a specific algorithm. One flag to consider always including is `--threads` with a value of `-1`. This will speed up execution in some cases but also might use all available cores. `--threads` with a value of `-2` will use all but one threads... etc.

The `--verbose` flag can be helpful to view data about a given dataset or MDP.

Each assignment folder should have its own readme with anything specific to not for that assignment.

## Why should I trust _you_, of all people?

You shouldn't.

## But a thing is broken!?

Feel free to open an issue for things that are flat out broken (or even better open a PR) and I can take a look.

That said, caveat emptor applies.

## Why didn't you fork Chad's or PR into his repo?

It's possible I may continue using Chad's stuff for the remainder of the assignments... but I don't know yet. Didn't want to restrict myself.
