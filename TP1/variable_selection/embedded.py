from typing import List
from copy import deepcopy

import numpy as np

from variable_selection.data import Data


class RecursiveFeatureElimination:
    def __init__(self, model, data: Data, feature_rank_attr: str):
        self.model = model
        self.data = deepcopy(data)
        self.feature_rank_attr = feature_rank_attr
        self.ranking = []

    def rank_variables(self, remove_amount: int):
        """
        Ranks all variables, removing remove_amount in each step.
        """
        while len(self.data.variable_names):
            self._remove_variables_(remove_amount)

        self.ranking.reverse()

    def _remove_variables_(self, remove_amount: int):
        """
        Removes the remove_amount least important variables, as estimated by
        estimate_error and moves them to ranking.
        """
        self.model.fit(self.data.train_values, self.data.train_results)

        variable_importance = np.array(getattr(self.model, self.feature_rank_attr))
        # Make variable_importance a non-negative number array of shape (n_variables).
        variable_importance **= 2
        if variable_importance.ndim > 1:
            variable_importance = variable_importance.sum(axis=0)
        
        # Indexes of the least important remove_amount variables, in increasing
        # order of importance.
        sorted_indexes = np.argsort(variable_importance)[:remove_amount]
        self.ranking += [self.data.variable_names[index] for index in sorted_indexes]
        self._delete_from_data_(sorted_indexes)

    def _delete_from_data_(self, indexes: List[int]):
        """Removes the selected indexes from the data."""
        self.data.variable_names = np.delete(self.data.variable_names, indexes)
        self.data.train_values = [
            np.delete(point, indexes) for point in self.data.train_values
        ]
