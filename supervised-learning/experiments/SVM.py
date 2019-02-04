import warnings

import numpy as np
import sklearn

import experiments
import learners


class SVMExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-1/blob/master/SVM.py
        samples = self._details.ds.features.shape[0]
        features = self._details.ds.features.shape[1]

        gamma_fracs = np.arange(1/features, 2.1, 0.2)
        tols = np.arange(1e-8, 1e-1, 0.01)
        C_values = np.arange(0.001, 2.5, 0.25)
        iters = [-1, int((1e6/samples)/.8)+1]

        best_params_linear = None
        if self._details.ds_best_params is not None and 'SVM_Linear' in self._details.ds_best_params: 
            best_params_linear = self._details.ds_best_params['SVM_Linear']
        best_params_rbf = None
        if self._details.ds_best_params is not None and 'SVM_RBF' in self._details.ds_best_params: 
            best_params_rbf = self._details.ds_best_params['SVM_RBF']

        # Linear SVM
        params = {'SVM__max_iter': iters, 'SVM__tol': tols, 'SVM__class_weight': ['balanced'],
                  'SVM__C': C_values}
        complexity_param = {'name': 'SVM__C', 'display_name': 'Penalty', 'values': np.arange(0.001, 2.5, 0.1)}

        iteration_details = {
            'x_scale': 'log',
            'params': {'SVM__max_iter': [2**x for x in range(12)]},
        }

        # RBF SVM
        if len(np.unique(self._details.ds.classes)) > 2:
            decision_functions = ['ovo']
        else:
            decision_functions = ['ovo', 'ovr']
        params = {'SVM__max_iter': iters, 'SVM__tol': tols, 'SVM__class_weight': ['balanced'],
                  'SVM__C': C_values,
                  'SVM__decision_function_shape': decision_functions, 'SVM__gamma': gamma_fracs}
        complexity_param = {'name': 'SVM__C', 'display_name': 'Penalty', 'values': np.arange(0.001, 2.5, 0.1)}

        learner = learners.SVMLearner(kernel='rbf')
        if best_params_rbf is not None:
            learner.set_params(**best_params_rbf)
        best_params = experiments.perform_experiment(
            self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner, 'SVM_RBF', 'SVM',
            params, complexity_param=complexity_param, seed=self._details.seed, iteration_details=iteration_details,
            best_params=best_params_rbf,
            threads=self._details.threads, verbose=self._verbose)

        of_params = best_params.copy()
        learner = learners.SVMLearner(kernel='rbf')
        if best_params_rbf is not None:
            learner.set_params(**best_params_rbf)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name, learner,
                                       'SVM_RBF_OF', 'SVM', of_params, seed=self._details.seed,
                                       iteration_details=iteration_details,
                                       best_params=best_params_rbf,
                                       threads=self._details.threads, verbose=self._verbose,
                                       iteration_lc_only=True)
