import numpy as np
import itertools
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Minimerror:
    """
    Implémentation généralisée de l'algorithme Minimerror (Buhot & Gordon, 1997).
    Compatible avec données binaires, bipolaires et continues.
    """

    def __init__(
        self,
        T: float = 1.0,
        learning_rate: float = 0.05,
        init_method: str = "hebb",   # "hebb" | "random"
        hebb_noise: float = 0.0,     # bruit ajouté après Hebb. bruit anti-minimum local (pour les problèmes combinatoires fortement symétriques)
        normalize_weights: bool = True,
        scale_inputs: bool = True  # doit-on standardiser les données ?
    ):
        self.T = T
        self.learning_rate = learning_rate
        self.init_method = init_method
        self.hebb_noise = hebb_noise
        self.normalize_weights = normalize_weights
        self.scale_inputs = scale_inputs

        self.w = None
        self.scaler = StandardScaler() if scale_inputs else None

        self.history = {
            'weights': [],
            "error": [],
            "cost": [],
            "T": [],
            "stabilities": []
        }
        self.stabilities_test = []

    # --------------------------------------------------
    # Utilitaires
    # --------------------------------------------------

    def _add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def _prepare_labels(self, y):
        """Force y ∈ {−1, +1}"""
        y = np.asarray(y)
        return np.where(y == 0, -1, y)

    # --------------------------------------------------
    # Initialisation
    # --------------------------------------------------

    def initialize_weights(self, X, y):
        Xb = self._add_bias(X)

        if self.init_method == "hebb":
            self.w = np.sum(Xb * y[:, None], axis=0)
            assert self.w.shape[0] == Xb.shape[1] # mesure de sécurité

            # bruit anti-minimum local
            if self.hebb_noise > 0:
                self.w += self.hebb_noise * np.random.randn(*self.w.shape)

        elif self.init_method == "random":
            self.w = 0.01 * np.random.randn(Xb.shape[1])

        else:
            raise ValueError("init_method must be 'hebb' or 'random'")

        if self.normalize_weights:
            self.w /= np.linalg.norm(self.w) + 1e-12  # Purement numérique pour éviter une division par zéro

    # --------------------------------------------------
    # Stabilité γ
    # --------------------------------------------------

    def compute_stability(self, X, y):
        Xb = self._add_bias(X)
        norm = np.linalg.norm(self.w) + 1e-12
        return y * (Xb @ self.w) / norm

    # --------------------------------------------------
    # Coût
    # --------------------------------------------------

    def compute_cost(self, X, y):
        gamma = self.compute_stability(X, y)
        return np.sum(0.5 * (1 - np.tanh(gamma / (2 * self.T))))

    # --------------------------------------------------
    # Gradient
    # --------------------------------------------------

    def compute_gradient(self, X, y):
        Xb = self._add_bias(X)
        gamma = self.compute_stability(X, y)

        z = gamma / (2 * self.T)
        t = np.tanh(z)
        factor = 1 - t**2   # dérivée stable

        grad = -(y[:, None] * Xb) * factor[:, None]
        grad = np.sum(grad, axis=0) / (4 * self.T)

        return grad

    def set_optimal_weights(self) -> np.ndarray:
        """Retourne les poids du meilleur modèle selon l'erreur de validation."""
        if not self.history.get('error'):
            # Si pas de validation, retourner les derniers poids
            return self.w

        # Trouver l'époque avec la plus petite erreur de validation
        best_epoch = np.argmin(self.history['error'])
        return self.history['weights'][best_epoch]

    def get_ea(self):
        """
        Retourne l'erreur d'apprentissage.

        Returns:
        --------
        int or None : Erreur d'apprentissage
        """
        best_epoch = np.argmin(self.history['error'])
        return self.history['error'][best_epoch]

    def get_eg(self, X_test, y_true):
        """
        Calcule l'erreur en généralisation.

        Parameters:
        -----------
        X_test : array (n_samples, n_features)
            Données de test SANS biais
        y_true : array (n_samples,)
            Vraies classes

        Returns:
        --------
        int : Nombre d'erreurs
        """
        y_pred = self.predict(X_test)
        # Nombre d'erreurs
        errors = np.sum(y_pred.flatten() != y_true.flatten())
        return errors
    # --------------------------------------------------
    # Entraînement
    # --------------------------------------------------

    def train(self, X , y, epochs=200, anneal=True,
        T_final=0.01):

        y = self._prepare_labels(y)

        if self.scale_inputs:
            X = self.scaler.fit_transform(X)

        self.initialize_weights(X, y)

        T0 = self.T

        error_hist = float('inf')

        for epoch in range(epochs):
            if anneal:  # Recuit déterministe (diminution de T)
                self.T = T0 - (T0 - T_final) * (epoch / epochs)

            grad = self.compute_gradient(X, y)
            self.w -= self.learning_rate * grad

            if self.normalize_weights:
                self.w /= np.linalg.norm(self.w) + 1e-12

            error = np.mean(self.predict(X) != y)
            cost = self.compute_cost(X, y)
            stabilities = self.compute_stability(X, y)
            if error > error_hist:
                self.learning_rate *= 0.9

            error_hist = error

            self.history["error"].append(error)
            self.history["cost"].append(cost)
            self.history["T"].append(self.T)
            self.history["weights"].append(self.w)
            self.history["stabilities"].append(stabilities)

            if epoch % 50 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Erreur = {error * len(y):.4f} | "
                    f"Coût = {cost:.3f} | "
                    f"T = {self.T:.3f}"
                )
        self.w = self.set_optimal_weights()
        return self.history

    # --------------------------------------------------
    # Prédiction
    # --------------------------------------------------

    def predict(self, X):
        if self.scale_inputs:
            X = self.scaler.transform(X)

        Xb = self._add_bias(X)
        y_pred = np.sign(Xb @ self.w)
        y_pred[y_pred == 0] = 1
        return y_pred
    
    def save_weights(self, filepath: str):
        """Sauvegarde les poids dans un fichier .npy"""
        np.savetxt(filepath, self.w, delimiter=',')


def plot_minimerror_pca(model, X, y, title="Minimerror – PCA + hyperplan"):
    """
    Visualisation PCA 2D :
    - points projetés
    - classes (+1 bleu, −1 rouge)
    - hyperplan projeté
    - couleur = stabilité γ
    """

    # -------- Préparation données --------
    y = np.where(y == 0, -1, y)

    if model.scale_inputs:
        X_scaled = model.scaler.transform(X)
    else:
        X_scaled = X.copy()

    gamma = model.compute_stability(X_scaled, y)

    # -------- PCA --------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # -------- Projection du vecteur normal --------
    w_no_bias = model.w[:-1]
    w_pca = pca.components_ @ w_no_bias
    b = model.w[-1]

    # -------- Grille pour l’hyperplan --------
    x_vals = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 300)

    if abs(w_pca[1]) < 1e-8:
        y_vals = np.zeros_like(x_vals)
    else:
        y_vals = -(w_pca[0] * x_vals + b) / w_pca[1]

    # -------- Plot --------
    plt.figure(figsize=(9, 7))

    # Classes
    pos = y == 1
    neg = y == -1

    plt.scatter(
        X_pca[pos, 0], X_pca[pos, 1],
        c=gamma[pos],
        cmap="Blues",
        edgecolor="black",
        label="Classe +1",
        s=70
    )

    plt.scatter(
        X_pca[neg, 0], X_pca[neg, 1],
        c=gamma[neg],
        cmap="Reds",
        edgecolor="black",
        label="Classe −1",
        s=70
    )

    # Hyperplan
    plt.plot(
        x_vals, y_vals,
        "k--",
        linewidth=2,
        label="Hyperplan (PCA)"
    )

    # Mise en forme
    plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
    plt.axvline(0, color="gray", linestyle=":", alpha=0.5)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    # Colorbar stabilité
    cbar = plt.colorbar()
    cbar.set_label("Stabilité γ")

    plt.tight_layout()
    plt.show()


def plot_cost_and_derivative(model):
    """
    Trace la fonction coût V(γ) et sa dérivée dV/dγ
    en fonction de γ/T sur l'intervalle [-2, 2].
    """

    # Température finale
    T = model.history["T"][-1]

    # Axe γ/T
    x = np.linspace(-2, 2, 500)
    gamma = x * T

    # Fonction coût
    V = 0.5 * (1 - np.tanh(gamma / (2 * T)))

    # Dérivée du coût
    dV = -1 / (4 * T) * (1 / np.cosh(gamma / (2 * T))**2)

    # Plot
    plt.figure(figsize=(8, 6))

    plt.plot(
        x, V,
        label="V(γ) — fonction coût",
        linewidth=2,
        color="black"
    )

    plt.plot(
        x, dV,
        label="dV/dγ — dérivée",
        linewidth=2,
        linestyle="--",
        color="red"
    )

    plt.axvline(0, color="gray", linestyle=":")
    plt.axhline(0, color="gray", linestyle=":")

    plt.xlabel(r"$\gamma / T$")
    plt.ylabel("Valeur")
    plt.title("Fonction coût Minimerror et dérivée (fin entraînement)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ======================================================================
# Jeu de données N-parité
# ======================================================================
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


# ======================================================================
# Test principal
# ======================================================================

if __name__ == "__main__":

    np.random.seed(0)

    n = 5  # 5
    dataset = NParityDataset(n)
    X, y = dataset.X, dataset.y
    print(X)

    model = Minimerror(
        init_method="hebb",
        normalize_weights=False,
        scale_inputs=False,
        hebb_noise=1e-2,
        learning_rate=0.1,
        T=100  # 15.0,
    )

    history = model.train(
        X, y,
        epochs=800,
        T_final=0.01
    )  # T_initial=50,  # 15.0

    final_error = model.history["error"][-1]

    print("\nRésultat final")
    print(f"Erreurs = {int(final_error * len(y))} / {len(y)}")

    print("\n Stabilité moyenne : ", np.mean(model.history["stabilities"][-1]))
    plot_minimerror_pca(model=model, X=X, y=y, title="Minimerror – PCA + hyperplan")

    plot_cost_and_derivative(model=model)

