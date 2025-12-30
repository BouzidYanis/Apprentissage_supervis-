from typing import Tuple
import itertools
import numpy as np


class NParityDataset:
    def __init__(self, n: int):
        self.n = n
        self.X, self.y = self.generate()

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        X = np.array(list(itertools.product([-1, 1], repeat=self.n)),
                     dtype=np.float32)
        y = np.array(
            [1 if np.sum(x == 1) % 2 == 0 else -1 for x in X],
            dtype=np.float32
        )
        return X, y


def theoretical_min_errors(N):
    """
    Calcule le nombre minimum d'erreurs théorique vf selon l'équation (4) du papier.
    vf = 2^(N-1) - Binomial(N-1, m)
    Où m = floor(N/2) (ou proche, voir papier pour cas pair/impair)
    [cite: 119, 147]
    """
    # Pour N=2p, m=p. Pour N=2p+1, m=p ou m=p+1.
    # On maximise le coefficient binomial, donc on prend m = floor((N-1)/2)
    # Note: Dans le papier m semble être défini par rapport aux sommets.
    # Pour simplifier, on cherche le coefficient binomial central de N-1.
    from scipy.special import comb

    m = (N - 1) // 2
    # Le papier indique qu'on soustrait le plus grand coefficient binomial
    # binomial_max correspond au milieu du triangle de Pascal pour la ligne N-1
    binomial_val = comb(N - 1, m)

    vf = 2 ** (N - 1) - binomial_val
    return int(vf)


class GenerateLS:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.perceptron_maitre = None

    @staticmethod
    def sign(h):
        return 1 if h >= 0 else -1

    @staticmethod
    def add_biais(X):
        P = X.shape[0]
        return np.c_[np.ones(P), X]

    def get_ls(self):
        # Perceptron maître
        self.perceptron_maitre = np.random.randn(self.n + 1)

        # Entrées centrées
        X = np.random.randn(self.p, self.n)
        Xb = self.add_biais(X)

        # Labels
        y = np.sign(Xb @ self.perceptron_maitre)
        y[y == 0] = 1

        return X, y