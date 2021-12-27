from operator import itemgetter
from typing import List

from scipy.stats import kruskal
from numpy import unique

from variable_selection.data import Data


class KruskalWallis:
    def __init__(self, data: Data):
        self.data = data
        self.remaining_variables = len(data.variable_names)
        self.h_statistic = []
        self._calculate_h_()

    def filter_variables(self, variable_amount: int):
        """Filters the variable_amount variables with the highest H statistic."""
        self.remaining_variables = variable_amount

    def filtered_variables(self) -> List[str]:
        """Returns the filtered variable names in decreasing order of importance."""
        return self.data.variable_names[: self.remaining_variables]
    
    def get_h_statistic(self) -> List[float]:
        """Returns the H statistic value of the variables."""
        return self.h_statistic[: self.remaining_variables]


    def _calculate_h_(self):
        """Calculates the H statistic for every variable in the training data."""
        variable_values = [
            [data_point[index] for data_point in self.data.train_values]
            for index, _ in enumerate(self.data.variable_names)
        ]

        self.h_statistic = []
        # Normalize classes to be [0, class_count).
        unique_klasses, klasses = unique(self.data.train_results, return_inverse=True)

        for values in variable_values:
            # Group by class.
            values_by_klass = [[] for _ in unique_klasses]
            for value, klass in zip(values, klasses):
                values_by_klass[klass].append(value)

            self.h_statistic.append(kruskal(*values_by_klass).statistic)

        self._sort_values_by_h_(variable_values)

    def _sort_values_by_h_(self, variable_values):
        """
        Sorts variable names by descending H statistic value.
        Reorders training data and h_statistic to match the new variable name ordering.
        """
        zipped_lists = zip(self.h_statistic, self.data.variable_names, variable_values)
        # Sort the zipped lists by h_statistic.
        sorted_tuples = sorted(zipped_lists, key=itemgetter(0), reverse=True)
        # Unpack the lists.
        sorted_lists = zip(*sorted_tuples)
        self.h_statistic, self.data.variable_names, variable_values = [
            list(l) for l in sorted_lists
        ]

        # Rearrange data.train_values
        for index in range(len(self.data.train_values)):
            self.data.train_values[index] = [
                variable_value[index] for variable_value in variable_values
            ]
