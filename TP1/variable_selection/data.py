from typing import Any, List
import random


class Data:
    def __init__(self, file_stem: str):
        self.file_stem = file_stem
        self.variable_names = []
        self.test_values = []
        self.test_results = []
        self.train_values = []
        self.train_results = []

    def read_train_data(self):
        """
        Reads train data from file "file_stem.data".
        """
        self.__read_data(
            f"{self.file_stem}.data", self.train_values, self.train_results
        )

    def read_test_data(self):
        """
        Reads test data from file "file_stem.test".
        """
        self.__read_data(f"{self.file_stem}.test", self.test_values, self.test_results)

    def make_test_from_train(self, test_ratio: float = 0.2):
        """
        Separates a random subsample from the train data to use as test data.
        The size of the test data is test_ratio * size of train data.
        """

        if test_ratio < 0 or test_ratio > 1:
            raise Exception("test_ratio needs to be a float between 0 and 1.")
        test_len = int(len(self.train_values) * test_ratio)

        # Random shuffle train data.
        train_data = list(zip(self.train_values, self.train_results))
        random.shuffle(train_data)
        self.train_values, self.train_results = zip(*train_data)

        # Separate test data.
        self.test_values = self.train_values[:test_len]
        self.test_results = self.train_results[:test_len]
        self.train_values = self.train_values[test_len:]
        self.train_results = self.train_results[test_len:]

    def __read_data(self, filename: str, values: List[Any], results: List[Any]):
        """
        Reads float values for the value list and a str for the result (a class).
        """
        file = open(filename, "r")
        values.clear()
        results.clear()

        self.variable_names = file.readline().replace('"', "").split()[:-1]
        for line in file.readlines():
            input_list = [float(value) for value in line.replace('"', "").split()]
            values.append(input_list[:-1])
            results.append(str(input_list[-1]))
