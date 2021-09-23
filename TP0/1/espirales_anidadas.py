import sys
from typing import List
from numpy import random, pi, sqrt, arctan2

if len(sys.argv) != 2:
    raise Exception("Usage: espirales_anidadas.py n")

point_amount = int(sys.argv[1])


def to_polar(x: float, y: float) -> List[float]:
    radius = sqrt(x * x + y * y)
    angle = arctan2(x, y)
    return [radius, angle]


class Spiral:
    def __init__(self, max_radius: float):
        self.max_radius = max_radius

    def __is_class_0(self, x: float, y: float):
        radius, angle = to_polar(x, y)
        base_curve = angle / (4 * pi)
        return (
            (radius >= base_curve and radius <= base_curve + 0.25)
            or (radius >= base_curve + 0.5 and radius <= base_curve + 0.75)
            or radius >= base_curve + 1
        )

    def __is_class_1(self, x: float, y: float) -> bool:
        return not self.__is_class_0(x, y)

    def __generate_point(self) -> List[float]:
        x, y = 1, 1
        while x * x + y * y > self.max_radius:
            x = random.rand() * 2 - 1
            y = random.rand() * 2 - 1
        return [x, y]

    def generate_point_class(self, klass: int) -> List[float]:
        x, y = self.__generate_point()
        if klass == 0:
            while not self.__is_class_0(x, y):
                x, y = self.__generate_point()
        elif klass == 1:
            while not self.__is_class_1(x, y):
                x, y = self.__generate_point()
        return [x, y]


generator = Spiral(1)
output_file = open("espirales_anidadas.data", "w")

for klass in range(2):
    for _i in range(point_amount // 2):
        output_file.write(
            ", ".join(map(str, generator.generate_point_class(klass))) + f", {klass}\n"
        )
