import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, eta=0.01, max_epoch=500, pocket=False, pocket_threshold=0):
        self.eta = eta  # learning rate
        self.max_epoch = max_epoch
        self.w = None
        self.w_pocket = None  # Meilleurs poids trouvés
        self.errors_history = []
        self.best_errors_history = []  # Historique des meilleures erreurs
        self.min_error = float('inf')  # Meilleure erreur rencontrée
        self.perceptron_maitre = None
        self.pocket = pocket
        self.pocket_threshold = pocket_threshold
        self.stabilities = None
        self.epoch = 0

    @staticmethod
    def sign(h):
        return 1 if h >= 0 else 0

    def initialize_perceptron_vectorized(self, x_train, y_train):
        """
        Version vectorisée de l'initialisation Hebb
        w_j = Σ_{μ=1}^p x_j^μ τ^μ = X^T · τ

        Le biais w_0 est initialisé séparément
        """
        # Convertir y en {-1, 1}
        tau = 2 * y_train.flatten() - 1  # τ^μ ∈ {-1, 1}

        # Calcul vectorisé pour les features: w_features = X^T · τ
        w_features = np.dot(x_train.T, tau)

        # Normalisation optionnelle (évite les poids trop grands)
        norm = np.linalg.norm(w_features)
        if norm > 0:
            w_features = w_features / norm

        # Initialisation du biais : généralement 0 ou petite valeur
        w_bias = 0.0  # CORRECTION: pas np.mean(tau)

        self.w = np.concatenate([[w_bias], w_features])
        return True

    def compute_errors(self, X, y, w):
        """Calcule le nombre d'erreurs avec les poids w"""
        # Version vectorisée pour la performance
        predictions = np.dot(X, w)
        y_pred = np.where(predictions >= 0, 1, 0)
        return np.sum(y_pred != y)

    def train_with_batch(self, x_train, y_train, init_method="random", shuffle=True):
        """
        Algorithme du perceptron avec option Pocket

        Args:
            x_train: données d'entraînement (sans biais)
            y_train: labels {0, 1}
            init_method: "random" ou "Hebb"
            shuffle: si True, mélange les données à chaque époque
        """
        if init_method not in ["random", "Hebb"]:
            raise ValueError("Les valeurs autorisées sont : 'random', 'Hebb'")

        # ajout du biais
        X = np.c_[np.ones(x_train.shape[0]), x_train]
        y = y_train.reshape(-1)

        # Initialisation des poids
        if init_method == 'Hebb':
            self.initialize_perceptron_vectorized(x_train, y_train)
        else:
            self.w = np.random.randn(X.shape[1])

        # Initialisation du Pocket
        self.w_pocket = self.w.copy()
        self.min_error = self.compute_errors(X, y, self.w)

        print(f"=========== Début apprentissage (Pocket={'activé' if self.pocket else 'désactivé'}) ===========")
        print(f"Erreur initiale: {self.min_error}/{len(y)}")

        for epoch in range(1, self.max_epoch + 1):
            # Mélange des indices si demandé
            indices = np.arange(len(X))
            if shuffle:
                np.random.shuffle(indices)

            errors = 0

            # Une époque d'apprentissage
            for i in indices:
                h = np.dot(X[i], self.w)
                y_pred = self.sign(h)

                if y_pred != y[i]:
                    self.w += self.eta * (y[i] - y_pred) * X[i]
                    errors += 1

            # Calcul de l'erreur APRÈS l'époque complète
            current_errors = self.compute_errors(X, y, self.w)

            # Mise à jour du Pocket si meilleure solution trouvée
            if self.pocket and current_errors < self.min_error:
                improvement = self.min_error - current_errors
                self.min_error = current_errors
                self.w_pocket = self.w.copy()
                if improvement > 0:  # Évite d'afficher si pas d'amélioration
                    print(f"Époque {epoch:3d}: Pocket amélioré ({current_errors} erreurs)")

            # Historique
            self.errors_history.append(errors)
            self.best_errors_history.append(self.min_error)

            # Critères d'arrêt
            stop_reason = None

            if self.min_error <= self.pocket_threshold:
                stop_reason = f"Critère Pocket atteint (erreur ≤ {self.pocket_threshold})"

            elif current_errors == 0:
                stop_reason = "Séparation linéaire parfaite atteinte"
                if self.pocket:
                    self.w_pocket = self.w.copy()
                    self.min_error = 0

            elif epoch == self.max_epoch:
                stop_reason = f"Nombre maximum d'époques atteint ({self.max_epoch})"

            # Arrêt si critère rempli
            if stop_reason:
                self.epoch = epoch
                print(f"\n=========== {stop_reason} ===========")
                print(f"Dernière époque: {epoch}")
                print(f"Meilleure erreur Pocket: {self.min_error}/{len(y)}")
                print(f"Erreur finale: {current_errors}/{len(y)}")
                break

        # Utiliser les meilleurs poids si Pocket activé
        if self.pocket:
            self.w = self.w_pocket.copy()
            print(f"Utilisation des poids Pocket (erreur: {self.min_error}/{len(y)})")

        print("=========== Apprentissage terminé ===========")
        return self.min_error

    def train_online(self, x_train, y_train, init_method="random", shuffle=True):
        """
        Algorithme du perceptron en ligne (stochastique) avec option Pocket

        Args:
            x_train: données d'entraînement (sans biais)
            y_train: labels {0, 1}
            init_method: "random" ou "Hebb"
            shuffle: si True, mélange les données à chaque époque
        """
        if init_method not in ["random", "Hebb"]:
            raise ValueError("Les valeurs autorisées sont : 'random', 'Hebb'")

        # Ajout du biais
        X = np.c_[np.ones(x_train.shape[0]), x_train]
        y = y_train.reshape(-1)  # vecteur 1D

        # Initialisation des poids
        if init_method == 'Hebb':
            self.initialize_perceptron_vectorized(x_train, y_train)
        else:  # random
            self.w = np.random.randn(X.shape[1])

        # Initialisation du Pocket si activé
        if self.pocket:
            self.w_pocket = self.w.copy()
            self.min_error = self.compute_errors(X, y, self.w)
            print(f"Erreur initiale Pocket: {self.min_error}/{len(y)}")
        else:
            self.min_error = float('inf')

        print(f"=========== Début apprentissage Online (Pocket={'activé' if self.pocket else 'désactivé'}) ===========")

        for epoch in range(self.max_epoch):
            errors = 0

            # Mélange aléatoire des indices pour l'époque
            indices = np.arange(len(X))
            if shuffle:
                indices = np.random.permutation(len(X))

            # Une époque d'apprentissage en ligne
            for i in indices:
                h = np.dot(X[i], self.w)
                y_pred = self.sign(h)

                if y_pred != y[i]:
                    # Mise à jour IMMÉDIATE (en ligne)
                    self.w += self.eta * (y[i] - y_pred) * X[i]
                    errors += 1

            # Calcul de l'erreur APRÈS l'époque
            current_errors = self.compute_errors(X, y, self.w)

            # Mise à jour du Pocket si meilleure solution trouvée
            if self.pocket and current_errors < self.min_error:
                improvement = self.min_error - current_errors
                self.min_error = current_errors
                self.w_pocket = self.w.copy()
                if improvement > 0:
                    print(f"Époque {epoch + 1:3d}: Pocket amélioré ({current_errors} erreurs)")

            # Historique
            self.errors_history.append(errors)
            if self.pocket:
                self.best_errors_history.append(self.min_error)
            else:
                self.best_errors_history.append(current_errors)

            # Critères d'arrêt
            stop_reason = None

            if self.pocket and self.min_error <= self.pocket_threshold:
                stop_reason = f"Critère Pocket atteint (erreur ≤ {self.pocket_threshold})"

            elif current_errors == 0:
                stop_reason = "Séparation linéaire parfaite atteinte"
                if self.pocket:
                    self.w_pocket = self.w.copy()
                    self.min_error = 0

            elif epoch == self.max_epoch - 1:
                stop_reason = f"Nombre maximum d'époques atteint ({self.max_epoch})"

            # Arrêt si critère rempli
            if stop_reason:
                self.epoch = epoch + 1
                print(f"\n=========== {stop_reason} ===========")
                print(f"Dernière époque: {epoch + 1}")
                if self.pocket:
                    print(f"Meilleure erreur Pocket: {self.min_error}/{len(y)}")
                print(f"Erreur finale: {current_errors}/{len(y)}")
                break

        # Si on sort sans break explicite
        if stop_reason is None:
            self.epoch = self.max_epoch
            print(f"\n=========== Arrêt après {self.max_epoch} époques ===========")
            print(f"Erreur finale: {current_errors}/{len(y)}")
            if self.pocket:
                print(f"Meilleure erreur Pocket: {self.min_error}/{len(y)}")

        # Utiliser les meilleurs poids si Pocket activé
        if self.pocket:
            self.w = self.w_pocket.copy()
            print(f"Utilisation des poids Pocket (erreur: {self.min_error}/{len(y)})")

        print("=========== Apprentissage Online terminé ===========")
        return current_errors

    def get_perceptron(self):
        return self.w.copy() if self.w is not None else None

    def predict(self, x_test):
        """
        Prédit les classes pour un ensemble de test
        X_test : array (p exemples, n features) SANS biais
        """
        if self.w is None:
            raise ValueError("Perceptron non entraîné. Appelez train_* d'abord.")

        # Ajout du biais
        x_test_with_bias = np.c_[np.ones(x_test.shape[0]), x_test]

        h = np.dot(x_test_with_bias, self.w)

        return np.array([Perceptron.sign(val) for val in h]).reshape(-1, 1)

    def get_ea(self):
        """
        Retourne l'erreur d'apprentissage moyenne (dernière époque)
        """
        if not self.min_error:
            return None
        # Si vous voulez la dernière erreur :
        return self.min_error

    def get_eg(self, X_test, y_true):
        """
        Calcul l'erreur en généralisation
        X_test : données de test SANS biais
        y_true : vraies classes
        """
        y_pred = self.predict(X_test)
        # Nombre d'erreurs
        errors = np.sum(y_pred.flatten() != y_true.flatten())
        return errors

    def compute_stability(self, X_data, y_data):
        """
        Calcule la stabilité (marge normalisée) pour chaque exemple
        Formule : γ^μ = (τ^μ * (w·x^μ)) / ||w||
        où τ^μ ∈ {-1, 1} mais nos classes sont {0, 1}

        X_data : array (n_samples, n_features) SANS biais
        y_data : array (n_samples,) classes {0, 1}
        """
        if self.w is None:
            raise ValueError("Perceptron non entraîné.")

        # Convertir y de {0, 1} à {-1, 1} pour la formule
        y_bipolar = 2 * y_data.flatten() - 1  # 0→-1, 1→+1

        # Ajouter biais aux données
        X_with_bias = np.c_[np.ones(X_data.shape[0]), X_data]

        # Produits scalaires w·x^μ
        h = np.dot(X_with_bias, self.w)

        # Calcul des stabilités : γ^μ = (τ^μ * h^μ) / ||w||
        norm_w = LA.norm(self.w, ord=2)  # Norme euclidienne
        if norm_w == 0:
            raise ValueError("Norme du vecteur poids nulle.")

        self.stabilities = (y_bipolar * h) / norm_w

        return self.stabilities.copy()

    def plot_stability_geometric(self, X_data, y_data, title="Représentation géométrique des stabilités"):
        """
        Représente géométriquement le perceptron (vecteur w) et chaque point
        selon sa distance/projection sur w.

        Pour N=2 (2 features) : représentation 2D
        Pour N>2 : projection sur les 2 premières dimensions OU PCA
        """
        if self.w is None:
            raise ValueError("Perceptron non entraîné.")

        # Calculer les stabilités
        stabilities = self.compute_stability(X_data, y_data)

        n_features = X_data.shape[1]

        if n_features == 2:
            self._plot_2d_geometric(X_data, y_data, stabilities, title)
        elif n_features == 1:
            self._plot_1d_geometric(X_data, y_data, stabilities, title)
        else:
            print(f"N={n_features} > 2, utilisation de PCA pour la visualisation")
            self._plot_pca_geometric(X_data, y_data, stabilities, title)

    def _plot_1d_geometric(self, X_data, y_data, stabilities, title):
        """Visualisation pour N=1 (1 feature)"""
        plt.figure(figsize=(10, 6))

        # Points colorés par stabilité
        scatter = plt.scatter(X_data[:, 0], np.zeros_like(X_data[:, 0]),
                              c=stabilities, cmap='coolwarm',
                              s=100, alpha=0.8, edgecolors='black')

        # Ligne du perceptron (droite verticale pour w0 + w1*x = 0)
        w0, w1 = self.w[0], self.w[1]
        if w1 != 0:
            x_decision = -w0 / w1
            plt.axvline(x=x_decision, color='red', linestyle='--',
                        linewidth=3, label=f'Frontière: x={x_decision:.2f}')

        # Flèche du vecteur w (projeté sur x)
        plt.arrow(0, -0.5, w1, 0, head_width=0.1, head_length=0.1,
                  fc='green', ec='green', linewidth=3,
                  label=f'w₁={w1:.2f} (composante x)')

        plt.colorbar(scatter, label='Stabilité γ')
        plt.xlabel('x₁')
        plt.title(f'{title} - N=1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([-1, 1])
        plt.show()

    def _plot_2d_geometric(self, X_data, y_data, stabilities, title):
        """Visualisation pour N=2 (2 features)"""
        fig = plt.figure(figsize=(14, 6))

        # Sous-graphique 1: Points et vecteur w
        ax1 = fig.add_subplot(121)

        # Points colorés par stabilité
        scatter = ax1.scatter(X_data[:, 0], X_data[:, 1],
                              c=stabilities, cmap='coolwarm',
                              s=100, alpha=0.8, edgecolors='black')

        # Vecteur w (sans le biais w0 pour la visualisation)
        w_features = self.w[1:]  # w1 et w2
        w_norm = np.linalg.norm(w_features)

        if w_norm > 0:
            # Normaliser pour une flèche de taille raisonnable
            scale = 2.0 / w_norm if w_norm > 0 else 1
            w_scaled = w_features * scale

            # Flèche du vecteur w
            ax1.arrow(0, 0, w_scaled[0], w_scaled[1],
                      head_width=0.2, head_length=0.3,
                      fc='green', ec='green', linewidth=3,
                      label=f'w=[{w_features[0]:.2f}, {w_features[1]:.2f}]')

            # Ligne perpendiculaire à w (frontière de décision)
            # w0 + w1*x + w2*y = 0 -> y = -(w0 + w1*x)/w2
            w0, w1, w2 = self.w[0], self.w[1], self.w[2]
            x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
            x_vals = np.linspace(x_min, x_max, 100)

            if abs(w2) > 1e-10:
                y_vals = -(w0 + w1 * x_vals) / w2
                ax1.plot(x_vals, y_vals, 'r--', linewidth=2,
                         label='Frontière w·x=0')

            # Projection des points sur w (ligne pointillée)
            for i, (x, y) in enumerate(X_data):
                point = np.array([x, y])
                # Projection sur w: (point·w) * w / ||w||²
                if w_norm > 0:
                    proj = np.dot(point, w_features) / (w_norm ** 2) * w_features
                    ax1.plot([x, proj[0]], [y, proj[1]], 'gray',
                             alpha=0.3, linewidth=0.5)

        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_title('Points et vecteur w')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Stabilité γ')

        # Sous-graphique 2: Distribution des projections sur w
        ax2 = fig.add_subplot(122)

        # Calculer les projections (distances signées)
        if w_norm > 0:
            projections = []
            for point in X_data:
                proj_dist = np.dot(point, w_features) / w_norm  # Distance signée
                projections.append(proj_dist)

            projections = np.array(projections)

            # Trier par projection pour un meilleur visuel
            sort_idx = np.argsort(projections)
            projections_sorted = projections[sort_idx]
            stabilities_sorted = stabilities[sort_idx]
            y_data_sorted = y_data.flatten()[sort_idx]

            # Points selon leur projection sur w
            for i, (proj, stab, y_true) in enumerate(zip(projections_sorted,
                                                         stabilities_sorted,
                                                         y_data_sorted)):
                color = 'blue' if y_true == 1 else 'red'
                marker = 'x' if stab < 0 else 'o'  # croix si mal classé
                ax2.scatter(proj, i, c=color, marker=marker, s=50, alpha=0.7)

            # Ligne à projection=0
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)

            ax2.set_xlabel('Projection sur w (distance signée)')
            ax2.set_ylabel('Index des exemples (triés)')
            ax2.set_title('Projection des points sur le vecteur w')
            ax2.grid(True, alpha=0.3)

            # Ajouter histogramme des projections
            ax2_hist = ax2.twinx()
            ax2_hist.hist(projections, bins=30, alpha=0.3, color='gray',
                          orientation='horizontal')
            ax2_hist.set_ylabel('Fréquence')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _plot_pca_geometric(self, X_data, y_data, stabilities, title):
        """Utilise PCA pour projeter en 2D quand N>2"""
        from sklearn.decomposition import PCA

        # Réduction à 2 dimensions avec PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_data)

        # Recalculer w dans l'espace PCA
        # w_pca = pca.components_.T @ self.w[1:]  # Approximatif

        fig = plt.figure(figsize=(12, 5))

        # Graphique 1: Points dans l'espace PCA
        ax1 = fig.add_subplot(121)
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1],
                              c=stabilities, cmap='coolwarm',
                              s=100, alpha=0.8, edgecolors='black')

        ax1.set_xlabel('PCA 1 ({:.1f}% variance)'.format(pca.explained_variance_ratio_[0] * 100))
        ax1.set_ylabel('PCA 2 ({:.1f}% variance)'.format(pca.explained_variance_ratio_[1] * 100))
        ax1.set_title('Projection PCA des données')
        plt.colorbar(scatter, ax=ax1, label='Stabilité γ')
        ax1.grid(True, alpha=0.3)

        # Graphique 2: Distance à la frontière vs stabilité
        ax2 = fig.add_subplot(122)

        # Calculer la "distance" approximative (produit scalaire)
        X_with_bias = np.c_[np.ones(X_data.shape[0]), X_data]
        distances = np.dot(X_with_bias, self.w) / np.linalg.norm(self.w[1:])

        ax2.scatter(distances, stabilities, c=y_data.flatten(),
                    cmap='viridis', s=80, alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2,
                    label='γ=0 (frontière)')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1,
                    label='w·x=0')

        ax2.set_xlabel('Distance signée à la frontière (w·x / ||w||)')
        ax2.set_ylabel('Stabilité γ')
        ax2.set_title('Stabilité vs distance à la frontière')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'{title} - PCA projection (N={X_data.shape[1]})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def afficher_graphique(self, X, y, name):
        # Vérifications
        if X.shape[1] > 2:
            print("Affichage disponible uniquement pour 2 caractéristiques (x1, x2).")
            return
        if self.w is None or len(self.w) != 3:
            print("Poids non initialisés ou de mauvaise dimension (attendu: 3 avec biais).")
            return

        # Aplatissement de y si nécessaire
        y_flat = y.reshape(-1)

        plt.figure(figsize=(6, 5))

        # Tracé des points par classe
        class0 = y_flat == 0
        class1 = y_flat == 1
        plt.scatter(X[class0, 0], X[class0, 1], color='red', marker='o', label='Classe 0')
        plt.scatter(X[class1, 0], X[class1, 1], color='blue', marker='x', label='Classe 1')

        # Droite de décision: w0 + w1*x1 + w2*x2 = 0 -> x2 = -(w0 + w1*x1)/w2
        w0, w1, w2 = self.w
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        x_vals = np.linspace(x_min - 0.5, x_max + 0.5, 200)

        plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

        if np.isclose(w2, 0.0):
            # Cas frontière verticale: w0 + w1*x1 = 0 -> x1 = -w0 / w1
            if not np.isclose(w1, 0.0):
                x_vert = -w0 / w1
                y_min, y_max = X[:, 1].min(), X[:, 1].max()
                plt.plot([x_vert, x_vert], [y_min, y_max], 'k--', label='Frontière de décision')
            else:
                print("Frontière non définie (w1 et w2 proches de 0).")
        else:
            y_vals = -(w0 + w1 * x_vals) / w2
            plt.plot(x_vals, y_vals, 'k--', label='Frontière de décision')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'frontière de décision de {name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class GenerateLS:
    def __init__(self, n, p):
        self.n = n
        self.p = p
        self.perceptron_maitre = None

    @staticmethod
    def sign(h):
        return 1 if h >= 0 else 0

    @staticmethod
    def add_biais(n, P):
        X = np.random.random((P, n))
        X = np.c_[np.ones(P), X]
        return X

    @staticmethod
    def remove_biais(X):
        return X[:, 1:]

    def get_ls(self):
        # Génération du perceptron maître
        self.perceptron_maitre = np.random.randn(self.n + 1)
        x_ls = GenerateLS.add_biais(self.n, self.p)  # array P x (n+1)
        y = np.zeros((self.p, 1))

        for i in range(self.p):
            h = np.dot(x_ls[i], self.perceptron_maitre)
            y[i] = GenerateLS.sign(h)

        return GenerateLS.remove_biais(x_ls), y
