import sys
from typing import List
from numpy import random, sqrt

if len(sys.argv) != 4:
    raise Exception("Usage: diagonal.py n d sigma")

point_amount = int(sys.argv[1])
dimension = int(sys.argv[2])
standard_dev = float(sys.argv[3])


class Gaussian:
    def __init__(self, standard_dev: float, center: List[float]):
        self.standard_dev = standard_dev
        self.center = center

    def generate_point(self) -> List[float]:
        return [
            self.__generate_coord(self.center[index])
            for index in range(len(self.center))
        ]

    def __generate_coord(self, center: float) -> float:
        return random.normal(loc=center, scale=self.standard_dev, size=1)[0]


class_0_generator = Gaussian(standard_dev, [-1] * dimension)
class_1_generator = Gaussian(standard_dev, [1] * dimension)

output_file = open("diagonal.data", "w")

for _i in range(point_amount // 2):
    output_file.write(", ".join(map(str, class_0_generator.generate_point())) + ", 0\n")

for _i in range(point_amount // 2):
    output_file.write(", ".join(map(str, class_1_generator.generate_point())) + ", 1\n")
