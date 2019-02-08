import argparse
from datetime import datetime
import logging
import numpy as np

import experiments
from data import loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(experiment_details, experiment, timing_key, verbose, timings):
    t = datetime.now()
    for details in experiment_details:
        exp = experiment(details, verbose=verbose)

        logger.info("Running {} experiment: {}".format(timing_key, details.ds_readable_name))
        exp.perform()
    t_d = datetime.now() - t
    if details.ds_name not in timings:
        timings[details.ds_name] = {}
        
    timings[details.ds_name][timing_key] = t_d.seconds
    # timings[timing_key] = t_d.seconds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform some SL experiments')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads (defaults to 1, -1 for auto)')
    parser.add_argument('--seed', type=int, help='A random seed to set, if desired')
    parser.add_argument('--ann', action='store_true', help='Run the ANN experiment')
    parser.add_argument('--boosting', action='store_true', help='Run the Boosting experiment')
    parser.add_argument('--dt', action='store_true', help='Run the Decision Tree experiment')
    parser.add_argument('--dtpost', action='store_true', help='Run the Decision Tree Post-Pruning experiment')
    parser.add_argument('--knn', action='store_true', help='Run the KNN experiment')
    parser.add_argument('--svm', action='store_true', help='Run the SVM experiment')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--verbose', action='store_true', help='If true, provide verbose output')
    args = parser.parse_args()
    verbose = args.verbose
    threads = args.threads

    seed = args.seed
    if seed is None:
        seed = np.random.randint(0, (2 ** 32) - 1)
        print("Using seed {}".format(seed))

    print("Loading data")
    print("----------")

    ds1_details = {
            'data': loader.CreditDefaultData(verbose=verbose, seed=seed),
            'name': 'credit_default',
            'best_params': None,
            'readable_name': 'Credit Default',
        }
    ds2_details = {
            'data': loader.PenDigitData(verbose=verbose, seed=seed),
            'name': 'pen_digits',
            'best_params': None,
            'readable_name': 'Handwritten Digits',
        }
    ds3_details = {
            'data': loader.StarcraftData(verbose=verbose, seed=seed),
            'name': 'starcraft_data',
            'best_params': None,
            'readable_name': 'Starcraft',
        }
    ds4_details = {
        'data': loader.CreditApprovalData(verbose=verbose, seed=seed),
        'name': 'credit_approval',
        'best_params': None,
        'readable_name': 'Credit Approval',
    }
    ds5_details = {
        'data': loader.AbaloneData(verbose=verbose, seed=seed),
        'name': 'abalone',
        'best_params': None,
        'readable_name': 'Abalone',
    }
    ds6_details = {
        'data': loader.HTRU2Data(verbose=verbose, seed=seed),
        'name': 'htru2',
        'best_params': None,
        'readable_name': 'HTRU2',
    }
    ds7_details = {
        'data': loader.SpamData(verbose=verbose, seed=seed),
        'name': 'spam',
        'best_params': None,
        'readable_name': 'SPAM',
    }
    ds8_details = {
        'data': loader.StatlogVehicleData(verbose=verbose, seed=seed),
        'name': 'statlog_vehicle_data',
        'best_params': None,
        'readable_name': 'Statlog Vehicle Data',
    }
    ds9_details = {
        'data': loader.WineData(verbose=verbose, seed=seed),
        'name': 'wine_quality',
        'best_params': None,
        'readable_name': 'Wine Quality',
    }
    ds10_details = {
        'data': loader.OnlineShoppersData(verbose=verbose, seed=seed),
        'name': 'online_shoppers_data',
        'best_params': None,
        'readable_name': 'Online Shoppers',
    }
    ds11_details = {
        'data': loader.AdultData(verbose=verbose, seed=seed),
        'name': 'adult_data',
        'best_params': None,
        'readable_name': 'Adult',
    }
    ds12_details = {
        'data': loader.MadelonData(verbose=verbose, seed=seed),
        'name': 'madelon_data',
        'best_params': None,
        'readable_name': 'Madelon',
    }
    ds12a_details = {
        'data': loader.MadelonCulled(verbose=verbose, seed=seed),
        'name': 'madelon_culled',
        'best_params': None,
        'readable_name': 'Madelon Culled',
    }
    ds13_details = {
        'data': loader.WallFollowing(verbose=verbose, seed=seed),
        'name': 'wall_following',
        'best_params': None,
        'readable_name': 'Wall-Following',
    }
    ds14_details = {
        'data': loader.StarcraftModified(verbose=verbose, seed=seed),
        'name': 'starcraft_modified',
        'best_params': None,
        'readable_name': 'StarCraft Mod',
    }


    if verbose:
        print("----------")
    print("Running experiments")

    timings = {}

    datasets = [
        ds1_details,
        ds2_details,
        ds3_details,
        ds4_details,
        # ds5_details, # needs tweaking
        ds6_details,
        ds7_details,
        ds8_details,
        # ds9_details, # needs tweaking
        ds10_details,
        ds11_details,
        ds12_details,
        ds12a_details,
        ds13_details,
        ds14_details
    ]

    experiment_details = []
    for ds in datasets:
        data = ds['data']
        data.load_and_process()
        data.build_train_test_split()
        data.scale_standard()
        data.create_histograms()
        experiment_details.append(experiments.ExperimentDetails(
            data, ds['name'], ds['readable_name'], ds['best_params'],
            threads=threads,
            seed=seed
        ))

    if args.ann or args.all:
        run_experiment(experiment_details, experiments.ANNExperiment, 'ANN', verbose, timings)

    if args.boosting or args.all:
        run_experiment(experiment_details, experiments.BoostingExperiment, 'Boosting', verbose, timings)

    if args.dt or args.all:
        run_experiment(experiment_details, experiments.DTExperiment, 'DT', verbose, timings)
    
    if args.dtpost:
        # This was an experiment that didn't work particularly well
        run_experiment(experiment_details, experiments.DTPostExperiment, 'DTPost', verbose, timings)

    if args.knn or args.all:
        run_experiment(experiment_details, experiments.KNNExperiment, 'KNN', verbose, timings)

    if args.svm or args.all:
        run_experiment(experiment_details, experiments.SVMExperiment, 'SVM', verbose, timings)

    print(timings)
