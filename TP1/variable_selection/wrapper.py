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
        self.end_state = []
        self.best_states = []
        self.best_states_error = []

    def select_variables(self, selection_size: int = -1):
        """
        Selects a subset of selection_size variables and trains the model with them.
        If no selection_size is given, there can be any amount of selected variables.
        """
        while self.state != self.end_state and sum(self.state) != selection_size:
            self.best_states_error.append(self._step_())
            self.best_states.append(self.state)
            

        if selection_size == -1:
            min_error = 1
            for state, error in zip(self.best_states, self.best_states_error):
                if error < min_error:
                    min_error = error
                    self.state = state
        else:
            for state in self.best_states:
                if sum(state) == selection_size:
                    self.state = state
                    break

        self._fit_model_()

    def selected_variables(self) -> List[str]:
        """Returns the name of the selected variables."""
        return [
            name for name, used in zip(self.data.variable_names, self.state) if used
        ]

    def _estimate_error_(self, variables: List[bool]) -> float:
        """
        Trains the model with the selected variables and returns the estimated error using 5-fold CV.
        """
        # If no variables are selected, return 1 as the error.
        if not any(variables):
            return 1

        return np.mean(
            cross_val_score(
                self.model,
                self._filter_variables_(self.data.train_values, variables),
                self.data.train_results,
            )
        )

    def _fit_model_(self) -> float:
        """Fits the model using the selected variables on all the train data."""
        return self.model.fit(
            self._filter_variables_(self.data.train_values, self.state),
            self.data.train_results,
        )

    def _filter_variables_(
        self, data: List[List[bool]], variables: List[bool]
    ) -> List[List[bool]]:
        """Returns a copy of the data with only the variables selected."""
        return [
            [value for value, used in zip(entry, variables) if used] for entry in data
        ]

    @abstractmethod
    def _step_(self):
        """
        Moves one step (adds/removes a variable from state).
        Chooses the variable that yields the model with the least estimated error.
        """
        pass


class ForwardWrapper(GreedyWrapper):
    def __init__(self, model, data: Data):
        super().__init__(model, data)
        self.state = [False] * len(data.variable_names)
        self.end_state = [True] * len(data.variable_names)

    def _step_(self) -> bool:
        min_error = 1
        best_step = self.state
        unused_vars = [index for index, used in enumerate(self.state) if not used]

        for unused_index in unused_vars:
            next_state = self.state.copy()
            next_state[unused_index] = True
            predicted_error = self._estimate_error_(next_state)
            if predicted_error < min_error:
                min_error = predicted_error
                best_step = next_state

        self.state = best_step
        return min_error


class BackwardWrapper(GreedyWrapper):
    def __init__(self, model, data: Data):
        super().__init__(model, data)
        self.state = [True] * len(data.variable_names)
        self.end_state = [False] * len(data.variable_names)

    def _step_(self) -> bool:
        min_error = 1
        best_step = self.state
        used_vars = [index for index, used in enumerate(self.state) if used]

        for used_index in used_vars:
            next_state = self.state.copy()
            next_state[used_index] = False
            predicted_error = self._estimate_error_(next_state)
            if predicted_error < min_error:
                min_error = predicted_error
                best_step = next_state

        self.state = best_step
        return min_error
