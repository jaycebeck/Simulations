import numpy as np


class Lorenz:
    def __init__(self, sigma=10, rho=28, beta=8 / 3):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def rhs(self, t, u):
        x, y, z = u
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return np.array([dxdt, dydt, dzdt])
