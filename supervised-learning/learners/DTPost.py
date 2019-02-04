import numpy as np
from sklearn import tree
import sklearn.model_selection as ms
from sklearn.tree._tree import TREE_LEAF
from copy import deepcopy
# import graphviz

import learners

def prune(clf, index, threshold):
    pruned_clf = deepcopy(clf)
    prune_index(pruned_clf.tree_, index, threshold)
    
    return pruned_clf

def prune_index(inner_tree, index, threshold):
    if threshold is not None:
        if (inner_tree.value[index].max() - inner_tree.value[index].sum()) < threshold:
            # turn node into a leaf by "unlinking" its children
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are shildren, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            prune_index(inner_tree, inner_tree.children_left[index], threshold)
            prune_index(inner_tree, inner_tree.children_right[index], threshold)

class DTPostLearner(learners.BaseLearner):
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False,
                 verbose=False):
        super().__init__(verbose)
        self.depth_threshold = max_depth
        self._learner = tree.DecisionTreeClassifier(
                 criterion=criterion,
                 splitter=splitter,
                 max_depth=None,
                 min_samples_split=min_samples_split,
                 min_samples_leaf=min_samples_leaf,
                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                 max_features=max_features,
                 random_state=random_state,
                 max_leaf_nodes=max_leaf_nodes,
                 min_impurity_decrease=min_impurity_decrease,
                 min_impurity_split=min_impurity_split,
                 class_weight=class_weight,
                 presort=presort)

    def learner(self):
        return self._learner

    @property
    def classes_(self):
        return self._learner.classes_

    @property
    def n_classes_(self):
        return self._learner.n_classes_
    
    def set_params(self, **params):
        """
        Set the current parameters for the learner. This passes the call back to the learner from learner()

        :param params: The params to set
        :return: self
        """
        #### Need to override the params to set max depth
        #### Need to export visualization
        self.depth_threshold = params['max_depth']
        params['max_depth'] = None
        return self.learner().set_params(**params)

    def fit(self, training, testing, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        print("fit params", self.get_params())
        print("depth threshold", self.depth_threshold)
        self._learner.fit(training, testing, sample_weight=sample_weight, check_input=check_input,
                                 X_idx_sorted=X_idx_sorted)
        self._learner = prune(self._learner, 0, self.depth_threshold)
        # self.write_visualization('./output/images/DTPost_statlog_vehicle_data_{}'.format(self.depth_threshold))
        return self._learner

    def write_visualization(self, path):
        """
        Write a visualization of the given learner to the given path (including file name but not extension)
        :return: self
        """
        tree.export_graphviz(self._learner, out_file="{}-{}.dot".format(path, self.depth_threshold), filled= True, rounded  = True, special_characters = True)
        with open("{}-{}.dot".format(path, self.depth_threshold)) as f:
            dot_graph = f.read()
        # graph = graphviz.Source(dot_graph)
        # graph_png = graph.pipe(format='png')
        # with open('{}-{}.png'.format(path, self.depth_threshold),'wb') as f:
        #     f.write(graph_png)

        return
