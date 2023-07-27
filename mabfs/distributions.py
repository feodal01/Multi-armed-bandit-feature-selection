import numpy as np


class BetaDistribution:

    def __init__(self, a: int, b: int):

        r"""
        :param a: beta shape parameter
        :param b: beta rate parameter
        """
        self.a = a
        self.b = b

    def sample(self) -> float:
        return np.random.beta(self.a, self.b)

    def update(self, result: int):
        self.a += result
        self.b += (1-result)
