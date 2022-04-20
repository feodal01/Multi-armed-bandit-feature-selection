import numpy as np


class BetaDistribution:

    def __init__(self, a: int, b: int):

        r"""
        :param a: beta shape parameter
        :param b: beta rate parameter
        """
        self.a = a
        self.b = b

    def sample(self) -> np.float:
        return np.random.beta(self.a, self.b)

    def update(self, result: int):
        self.a += result
        self.b += (1-result)
        
class GammaDistribution:

    def __init__(self, a: int, b: int):

        r"""
        :param a: gamma shape parameter = кумулятивный спрос за все периоды для этой цены
        :param b: gamma rate parameter = количество периодов в которых устанавливалась цена
        """
        self.a = a
        self.b = b

    def sample(self) -> np.float:
        return np.random.gamma(self.a, 1 / self.b)

    def update(self, demand_t: int):

        r"""
        In step 1 of TS-fixed, the posterior
        distribution of dik is Gamma(Wik(t−1) + 1, Nk(t−1) + 1), so we sample dik(t) independently from
        a Gamma(Wik(t − 1) + 1, Nk(t − 1) + 1) distribution for each price k and each product i.
        In steps 2 and 3, LP(d(t)) is solved and the price vector P(t) = pk0 for some k
        0 ∈ [K] is chosen;
        then the customer demand Di(t) is revealed to the retailer.

        In step 4, we then update Nk0(t) ← Nk0(t − 1) + 1, Wik0(t) ← Wik0(t − 1) + Di(t) for all i ∈ [N].
        The posterior distributions associated with the K − 1 unchosen price vectors (k 6= k0) are not changed.
        """

        self.a = max(self.a + demand_t, 1)
        self.b = self.b + 1
