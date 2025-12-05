import numpy as np


class Perceptron:
    def __init__(self, eta=0.01, max_epoch=500):
        self.eta = eta  # learning rate
        self.max_epoch = max_epoch
        self.w = None
        self.errors_history = []
        self.perceptron_maitre = None

    @staticmethod
    def sign(h):
        return 1 if h >= 0 else 0

    def train_with_batch(self, x_train, y_train):
        # Ajout du biais (colonne de 1)
        self.x = np.c_[np.ones(x_train.shape[0]), x_train]
        self.y = y_train.reshape(-1, 1)

        # Initialisation des poids (w0 pour biais, w1...wn pour features)
        self.w = np.random.randn(self.x.shape[1], 1)

        print("=========== Training in process ===========")

        for epoch in range(1, self.max_epoch + 1):
            errors = 0

            for i in range(len(self.x)):
                # Calcul de la sortie
                h = np.dot(self.x[i], self.w)
                y_pred = self.sign(h)

                if y_pred != self.y[i]:
                    # Mise à jour des poids
                    self.w += self.eta * (self.y[i] - y_pred) * self.x[i].reshape(-1, 1)
                    errors += 1

            self.errors_history.append(errors)
            print(f"Epoch {epoch}: errors = {errors}")

            if errors == 0:
                print(f"=========== Apprentissage réussi! ===========")
                print(f"Nombre d'epochs: {epoch}")
                break

        if errors > 0:
            print(f"=========== Arrêt après {self.max_epoch} epochs ===========")
            print(f"Dernier nombre d'erreurs: {errors}")

    def train_online(self, x_train, y_train):
        X = np.c_[np.ones(x_train.shape[0]), x_train]  # biais
        y = y_train.astype(int)

        # Init poids
        self.w = np.random.randn(X.shape[1])

        for epoch in range(self.max_epoch):
            errors = 0

            indices = np.random.permutation(len(X))

            for i in indices:
                h = np.dot(X[i], self.w)
                y_pred = self.sign(h)

                if y_pred != y[i]:
                    self.w += self.eta * y[i] * X[i]
                    errors += 1

            self.errors_history.append(errors)

            if errors == 0:
                print(f"Convergence à l’epoch {epoch + 1}")
                break
        if errors > 0:
            print(f"=========== Arrêt après {self.max_epoch} epochs ===========")
            print(f"Dernier nombre d'erreurs: {errors}")

    @staticmethod
    def add_biais(n, P):
        X = []
        for i in range(P):
            x = [1]
            for j in range(n):
                x.append(np.random.random())
            X.append(x)
        return X

    @staticmethod
    def remove_biais(x):
        for i in range(len(x)):
            x[i].pop(0)
        return x

    """
        1/ Données LS aléatoires. Construire un ensemble LS de P exemples en N+1
        dimensions avec un perceptron professeur W*. Attention : le poids W*(0) étant le biais,
        donc X(mu,0)=1

    """

    @staticmethod
    def generate_LS(n, p=20):
        perceptron_maitre = np.random.randn(n + 1, 1)
        x_ls = Perceptron.add_biais(n, p)
        y = [0]*p

        for i in range(p):
            h = np.dot(x_ls[i], perceptron_maitre)
            y[i] = Perceptron.sign(h)

        return np.array(Perceptron.remove_biais(x_ls)), np.array(y)
