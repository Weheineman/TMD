from operator import itemgetter
from typing import List, Tuple

from scipy.stats import kruskal

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

    def filtered_variables(self) -> List[Tuple[str, float]]:
        """Returns the filtered variable names with their H statistic value."""
        return list(
            zip(self.data.variable_names[: self.remaining_variables], self.h_statistic)
        )

    def _calculate_h_(self):
        """Calculates the H statistic for every variable in the training data."""
        if self.h_statistic:
            return

        variable_values = [
            [data_point[index] for data_point in self.data.train_values]
            for index in range(len(self.data.variable_names))
        ]

        self.h_statistic = [
            kruskal(values, self.data.train_results).statistic
            for values in variable_values
        ]
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