import numpy as np

import experiments
import learners


class DTPostExperiment(experiments.BaseExperiment):
    def __init__(self, details, verbose=False):
        super().__init__(details)
        self._verbose = verbose

    def perform(self):
        # TODO: Clean up the older alpha stuff?
        max_depths = np.arange(1, 51, 1)
        params = {'DTPost__criterion': ['entropy'], 'DTPost__max_depth': max_depths,
                  'DTPost__class_weight': [None]}  # , 'DTPost__max_leaf_nodes': max_leaf_nodes}
        complexity_param = {'name': 'DTPost__max_depth', 'display_name': 'Threshold', 'values': max_depths}

        best_params = None
        if self._details.ds_best_params is not None and 'DT' in self._details.ds_best_params: 
            best_params = self._details.ds_best_params['DT']
        learner = learners.DTPostLearner(random_state=self._details.seed)
        if best_params is not None:
            learner.set_params(**best_params)
        experiments.perform_experiment(self._details.ds, self._details.ds_name, self._details.ds_readable_name,
                                       learner, 'DTPost', 'DTPost', params,
                                       complexity_param=complexity_param, seed=self._details.seed,
                                       threads=self._details.threads,
                                       best_params=best_params,
                                       verbose=self._verbose)
