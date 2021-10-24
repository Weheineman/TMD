from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.model_selection import cross_val_score

from variable_selection.data import Data


class GreedyWrapper(ABC):
    def __init__(self, model, data: Data):
        self.model = model
        self.data = data
        self.state = []

    def select_variables(self):
        """Selects a subset of variables and trains the model with them."""
        while self._step():
            pass

        self._fit_model()

    def selected_variables(self) -> List[str]:
        """Returns the name of the selected variables."""
        return [
            name for name, used in zip(self.data.variable_names, self.state) if used
        ]

    def _estimate_error(self, variables: List[bool]) -> float:
        """
        Trains the model with the selected variables and returns the estimated error using 5-fold CV.
        """
        # If no variables are selected, return 1 as the error.
        if not any(variables):
            return 1

        return np.mean(
            cross_val_score(
                self.model,
                self._filter_variables(self.data.train_values, variables),
                self.data.train_results,
            )
        )

    def _fit_model(self) -> float:
        """Fits the model using the selected variables on all the train data."""
        return self.model.fit(
            self._filter_variables(self.data.train_values, self.state),
            self.data.train_results,
        )

    def _filter_variables(
        self, data: List[List[bool]], variables: List[bool]
    ) -> List[List[bool]]:
        """Returns a copy of the data with only the variables selected."""
        return [
            [value for value, used in zip(entry, variables) if used] for entry in data
        ]

    @abstractmethod
    def _step(self) -> bool:
        """
        Returns True if adding a variable improves the error, False otherwise.
        Updates the state and predicted_error if a step is taken.
        """
        pass


class ForwardWrapper(GreedyWrapper):
    def __init__(self, model, data: Data):
        super().__init__(model, data)
        self.state = [False] * len(data.variable_names)
        self.predicted_error = 1

    def _step(self) -> bool:
        lowest_error = self.predicted_error
        best_step = self.state
        unused_vars = [index for index, used in enumerate(self.state) if not used]

        for unused_index in unused_vars:
            next_state = self.state.copy()
            next_state[unused_index] = True
            predicted_error = self._estimate_error(next_state)
            if predicted_error < lowest_error:
                lowest_error = predicted_error
                best_step = next_state

        # If there is an improvement, it updates the state and returns True.
        if lowest_error < self.predicted_error:
            self.state = best_step
            self.predicted_error = lowest_error
            return True

        return False


class BackwardWrapper(GreedyWrapper):
    def __init__(self, model, data: Data):
        super().__init__(model, data)
        self.state = [True] * len(data.variable_names)
        self.predicted_error = self._estimate_error(self.state)

    def _step(self) -> bool:
        lowest_error = self.predicted_error
        best_step = self.state
        used_vars = [index for index, used in enumerate(self.state) if used]

        for used_index in used_vars:
            next_state = self.state.copy()
            next_state[used_index] = False
            predicted_error = self._estimate_error(next_state)
            if predicted_error < lowest_error:
                lowest_error = predicted_error
                best_step = next_state

        # If there is an improvement, it updates the state and returns True.
        if lowest_error < self.predicted_error:
            self.state = best_step
            self.predicted_error = lowest_error
            return True

        return False
