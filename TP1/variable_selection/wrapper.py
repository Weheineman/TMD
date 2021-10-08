from abc import ABC, abstractmethod
from typing import List

from sklearn.model_selection import cross_val_score

from data import Data

class GreedyWrapper(ABC):
    def __init__(self, model, data: Data):
        self.model = model
        self.data = data
        self.state = []

    def select_variables(self):
        """Selects a subset of variables and trains the model with them."""
        while self.__step():
            pass

        self.__fit_model()

    def __estimate_error(self, variables: List[bool]) -> float:
        """
        Trains the model with the selected variables and returns the estimated error using 5-fold CV.
        """
        return cross_val_score(
            self.model,
            self.__filter_variables(self.data.train_values, variables),
            self.data.train_results,
        )

    def __fit_model(self) -> float:
        """Fits the model using the selected variables on all the train data."""
        return self.model.fit(
            self.__filter_variables(self.data.train_values, self.state),
            self.data.train_results,
        )

    # TODO: Maybe always apply to train_values?
    def __filter_variables(
        self, data: List[List[bool]], variables: List[bool]
    ) -> List[List[bool]]:
        """Returns a copy of the data with only the variables selected."""
        return [
            [value for value, used in zip(entry, variables) if used] for entry in data
        ]

    @abstractmethod
    def __step(self):
        """
        Returns True if adding a variable improves the error, False otherwise.
        Updates the state and predicted_error if a step is taken.
        """
        pass


class ForwardWrapper(GreedyWrapper):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.state = [False] * data.dimension
        self.predicted_error = 1

    def __step(self) -> bool:
        lowest_error = self.predicted_error
        best_step = self.state
        unused_vars = [
            index for index, used in enumerate(self.current_state) if not used
        ]

        for unused_index in unused_vars:
            next_state = self.current_state.copy()
            next_state[unused_index] = True
            predicted_error = self.__estimate_error(next_state)
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
    def __init__(self, model, data):
        super().__init__(model, data)
        self.state = [True] * data.dimension
        self.predicted_error = self.__estimate_error(self.state)

    def __step(self) -> bool:
        lowest_error = self.predicted_error
        best_step = self.state
        used_vars = [index for index, used in enumerate(self.current_state) if used]

        for used_index in used_vars:
            next_state = self.current_state.copy()
            next_state[used_index] = False
            predicted_error = self.__estimate_error(next_state)
            if predicted_error < lowest_error:
                lowest_error = predicted_error
                best_step = next_state

        # If there is an improvement, it updates the state and returns True.
        if lowest_error < self.predicted_error:
            self.state = best_step
            self.predicted_error = lowest_error
            return True

        return False