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
        self.ranking = [None] * len(data.variable_names)

    def select_variables(self, selection_size: int = -1):
        """
        Selects a subset of selection_size variables and trains the model with them.
        If no selection_size is given, there can be any amount of selected variables.
        """
        while self.state != self.end_state and sum(self.state) != selection_size:
            self.best_states_error.append(self._step_())
            self.best_states.append(self.state)

        if selection_size == -1:
            min_error = 2
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

        # The error is 1 - the score.
        return 1 - np.mean(
            cross_val_score(
                self.model,
                self._filter_variables_(variables),
                self.data.train_results,
            )
        )

    def _fit_model_(self) -> float:
        """Fits the model using the selected variables on all the train data."""
        return self.model.fit(
            self._filter_variables_(self.state),
            self.data.train_results,
        )

    def _filter_variables_(self, variables: List[bool]) -> List[List[float]]:
        """Returns a copy of the data with only the variables selected."""
        return [
            [value for value, used in zip(entry, variables) if used]
            for entry in self.data.train_values
        ]

    def _step_(self):
        """
        Moves one step (adds/removes a variable from state).
        Chooses the variable that yields the model with the least estimated error.
        """
        min_error = 2
        next_vars = self._get_next_vars_()

        for next_index in next_vars:
            next_state = self.state.copy()
            self._set_next_state_(next_state, next_index)
            predicted_error = self._estimate_error_(next_state)
            if predicted_error < min_error:
                min_error = predicted_error
                best_step = next_state
                chosen_variable = next_index

        self._add_to_ranking_(chosen_variable)
        self.state = best_step
        return min_error

    @abstractmethod
    def _add_to_ranking_(self, index: int):
        """Adds the selected variable to the ranking."""
        pass

    @abstractmethod
    def _get_next_vars_(self) -> List[bool]:
        """Returns a list of the variables to be considered on the next step."""
        pass

    @abstractmethod
    def _set_next_state_(self, next_state: List[bool], index: int):
        """Modifies the next state at the selected index."""
        pass


class ForwardWrapper(GreedyWrapper):
    def __init__(self, model, data: Data):
        super().__init__(model, data)
        self.state = [False] * len(data.variable_names)
        self.end_state = [True] * len(data.variable_names)

    def _get_next_vars_(self) -> List[bool]:
        return [index for index, used in enumerate(self.state) if not used]

    def _set_next_state_(self, next_state: List[bool], index: int):
        next_state[index] = True

    def _add_to_ranking_(self, index: int):
        self.ranking[sum(self.state)] = self.data.variable_names[index]


class BackwardWrapper(GreedyWrapper):
    def __init__(self, model, data: Data):
        super().__init__(model, data)
        self.state = [True] * len(data.variable_names)
        self.end_state = [False] * len(data.variable_names)

    def _get_next_vars_(self) -> List[bool]:
        return [index for index, used in enumerate(self.state) if used]

    def _set_next_state_(self, next_state: List[bool], index: int):
        next_state[index] = False

    def _add_to_ranking_(self, index: int):
        self.ranking[sum(self.state) - 1] = self.data.variable_names[index]
