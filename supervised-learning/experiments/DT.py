import numpy as np

import experiments
import learners


class DTExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # TODO: Clean up the older alpha stuff?
        max_depths = np.arange(1, 51, 1)
        params = {'DT__criterion': ['gini', 'entropy'], 'DT__max_depth': max_depths,
                  'DT__class_weight': ['balanced', None]}  # , 'DT__max_leaf_nodes': max_leaf_nodes}
        complexity_param = {'name': 'DT__max_depth', 'display_name': 'Max Depth', 'values': max_depths}

        best_params = None
        learner = learners.DTLearner(random_state=self._details.seed)
        if self._details.ds_best_params is not None and 'DT' in self._details.ds_best_params: 
            best_params = self._details.ds_best_params['DT']
        if best_params is not None:
            learner.set_params(**best_params)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       learner, 'DT', 'DT', params,
                                       complexity_param=complexity_param, seed=self._details.seed,
                                       threads=self._details.threads,
                                       best_params=best_params,
                                       verbose=self._verbose)
