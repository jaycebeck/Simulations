import numpy as np


class Chaos:
    def __init__(self):
        pass

    def generate(self, r, x, iterations):
        x_values = []
        for i in range(iterations + 1):
            x_values.append(x)
            x = self.map(r, x)
        return x_values

    def map(self, r, x):
        return 4 * r * x * (1 - x)
