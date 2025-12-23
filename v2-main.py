import numpy as np
from tools import load_breast_cancer_data, parse_sonar_file
from v2 import Minimerror, MinimerrorTwoTemp, NParityDataset
import matplotlib.pyplot as plt
import pandas as pd


def n_parity(n):  # 5  7 pour T= 10

    np.random.seed(0)
    dataset = NParityDataset(n)
    X, y = dataset.X, dataset.y
    # print(X)

    model = Minimerror(
        T=0.5,  # Température initiale
        learning_rate=0.001,  # Learning rate adaptatif
        init_method='hebb',  # Meilleure initialisation
        hebb_noise=0.001,  # Léger bruit
        normalize_weights=True,
        scale_inputs=True,  # Important
        momentum=0.9,  # Accélération
        min_lr_ratio=0.001  # LR ne va pas à 0
    )

    history = model.train(
        X, y,
        epochs=4000,
        T_final=0.01
    )  # T_initial=50,  # 15.0

    best_epoch = np.argmin(model.history['error'])
    final_error = model.history["error"][best_epoch]

    print("\nRésultat final")
    print(f"Erreurs = {int(final_error * len(y))} / {len(y)}")

    print("\n Stabilité moyenne : ", np.mean(model.history["stabilities"][-1]))
    print("Weights : ", model.w)
    # plot_minimerror_pca(model=model, X=X, y=y, title="Minimerror – PCA + hyperplan")


# n_parity(7)


file_path = r"./data"
x_train, y_train = None, None
x_test, y_test = None, None
x_all, y_all = None, None


def init_2():
    global file_path
    global x_train, y_train
    global x_test, y_test
    global x_all, y_all

    # train, test sets for rocks
    filename = "sonar.mines"
    try:
        x_train_mines, x_test_mines, y_train_mines, y_test_mines = parse_sonar_file(filename, file_path)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filename}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")

    # train, test sets for rocks
    filename = "sonar.rocks"
    try:
        x_train_rocks, x_test_rocks, y_train_rocks, y_test_rocks = parse_sonar_file(filename, file_path)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{filename}' n'a pas été trouvé.")
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {e}")

    x_train = pd.concat([x_train_mines, x_train_rocks])
    x_test = pd.concat([x_test_mines, x_test_rocks])

    y_train = pd.concat([y_train_mines, y_train_rocks])
    y_test = pd.concat([y_test_mines, y_test_rocks])

    x_all = pd.concat([x_train, x_test])
    y_all = pd.concat([y_train, y_test])

    print(f"Dimension x_tain :({x_train.shape[0]}, {x_train.shape[1]})")

    print(f"Dimension x_test :({x_test.shape[0]}, {x_test.shape[1]})")

    print(f"Dimension x_all :({x_all.shape[0]}, {x_all.shape[1]})")

    print(y_test['class'].unique(), y_all.shape)

    # Il faut transformer les données catégorielles  M, R en 1, 0 pour les questions suivantes
    y_train['class'] = y_train['class'].map({'M': 1, 'R': 0})
    y_test['class'] = y_test['class'].map({'M': 1, 'R': 0})
    y_all['class'] = y_all['class'].map({'M': 1, 'R': 0})


def init_3():
    global x_train, y_train
    global x_test, y_test

    # Numpy arrays
    X_train = x_train.to_numpy().astype(float)
    y_train = y_train["class"].to_numpy().astype(float)

    x_test = x_test.to_numpy().astype(float)
    y_test = y_test["class"].to_numpy().astype(float)

    # Entraînement perceptron sur l'ensemble L
    model = Minimerror(
        T=100,  # Température initiale
        learning_rate=0.002,  # Learning rate adaptatif
        init_method='hebb',  # Meilleure initialisation
        hebb_noise=0,  # Léger bruit
        normalize_weights=True,
        scale_inputs=True,  # Important
        momentum=0.98,  # Accélération
        min_lr_ratio=0.0001  # LR ne va pas à 0
    )

    # Entraînement avec la NOUVELLE méthode
    history = model.train(
        X_train, y_train,
        epochs=5000,  # Suffisant
        anneal=True,
        T_final=0.05,  # Température finale basse
        gradient_method='auto',  # NOUVEAU : sélection intelligente
        early_stopping=True,  # Arrête si Ea = 0
        verbose=True
    )

    best_epoch = np.argmin(model.history['error'])
    Ea = model.history['error'][best_epoch]

    if Ea > 0:
        print("\nForçage de la convergence parfaite...")
        model.force_perfect_separation(X_train, y_train)

    y_pred_train = model.predict(X_train)
    Ea = np.sum(y_pred_train != y_train)

    y_pred_test = model.predict(x_test)
    Eg = np.sum(y_pred_test != y_test)

    print(f"Ea = {Ea}/{len(y_train)}")
    print(f"Eg = {Eg}/{len(y_test)}")

    print("\n Stabilité moyenne : ", np.mean(model.history["stabilities"][best_epoch]))
    # plot_minimerror_pca(model=model, X=Xn, y=yn, title="Minimerror – PCA + hyperplan")

    # plot_cost_and_derivative(model=model)
    # model.save_weights("minimerror_sonar_weights.csv")
# init_2()
# init_3()

# ======================================================================
# Test principal
# ======================================================================

# init_1()

# n_parity(n=6)
# init_2()
# init_3()


#  load breast_cancer_data
X_train, X_test, y_train, y_test = load_breast_cancer_data(test_size=0.2, random_state=42)


def parti_I():
    global X_train, X_test, y_train, y_test
    # Split manuel

    # CONFIGURATION OPTIMALE POUR Ea = 0
    model = Minimerror(
        T=0.5,  # Température initiale
        learning_rate=0.02,  # Learning rate adaptatif
        init_method='hebb',  # Meilleure initialisation
        hebb_noise=0.01,  # Léger bruit
        normalize_weights=True,
        scale_inputs=True,  # Important
        momentum=0.9,  # Accélération
        min_lr_ratio=0.001  # LR ne va pas à 0
    )

    # Entraînement avec la NOUVELLE méthode
    history = model.train(
        X_train, y_train,
        epochs=5000,  # Suffisant
        anneal=True,
        T_final=0.05,  # Température finale basse
        gradient_method='auto',  # NOUVEAU : sélection intelligente
        early_stopping=True,  # Arrête si Ea = 0
        verbose=True
    )

    y_pred_train = model.predict(X_train)  # as-t-on atteint la convergence parfaite ?

    Ea = np.sum(y_pred_train != y_train)

    # Si Ea n'est pas encore 0, forcer
    if Ea > 0:
        print("\nForçage de la convergence parfaite...")
        model.force_perfect_separation(X_train, y_train)

    y_pred_train = model.predict(X_train)
    Ea = np.sum(y_pred_train != y_train)

    y_pred_test = model.predict(X_train)
    Eg = np.sum(y_pred_test != y_test)

    print(f"Ea = {Ea}/{len(y_train)}")
    print(f"Eg = {Eg}/{len(y_test)}")

    pass


parti_I()

def parti_II():
    def run_experiment_two_temp(X_train, y_train, X_test, y_test,
                                beta_plus=1.0, beta_minus=1.0,
                                ratio_fixed=False, filename="weights.txt"):
        """
        Exécute l'expérience complète de la partie II.

        Parameters:
        -----------
        ratio_fixed : bool
            Si True, le rapport beta_plus/beta_minus est fixé à 1
            et on fait varier les deux simultanément.
        """
        print("=" * 60)
        print(f"EXPÉRIENCE AVEC β+={beta_plus}, β-={beta_minus}")
        print("=" * 60)

        # 1. Entraînement
        model = MinimerrorTwoTemp(
            beta_plus=beta_plus,
            beta_minus=beta_minus,
            learning_rate=0.05,
            init_method="hebb",
            normalize_weights=True
        )

        history = model.train(X_train, y_train, epochs=300, verbose=True)

        # 2. Calcul des erreurs
        Ea, Eg = model.compute_errors(X_train, y_train, X_test, y_test)
        print(f"\nEa (erreur apprentissage) = {Ea:.4f}")
        print(f"Eg (erreur généralisation) = {Eg:.4f}")

        # 3. Sauvegarde des poids
        model.save_weights(filename)

        # 4. Calcul des stabilités sur test
        stabilities_test = model.compute_test_stabilities(X_test)
        print(f"Stabilités test: moy={np.mean(stabilities_test):.3f}, "
              f"std={np.std(stabilities_test):.3f}")

        return model, Ea, Eg, stabilities_test

    def plot_stabilities_vs_beta(X_train, y_train, X_test, y_test,
                                 beta_values=range(1, 11)):
        """
        Graphique des stabilités en fonction de β (avec rapport fixé à 1).
        """
        all_stabilities = []
        all_Ea = []
        all_Eg = []

        print("\n" + "=" * 60)
        print("ÉTUDE DES STABILITÉS EN FONCTION DE β")
        print("=" * 60)

        for beta in beta_values:
            print(f"\nβ = {beta}")

            # Entraînement avec beta_plus = beta_minus = beta
            model, Ea, Eg, stabilities = run_experiment_two_temp(
                X_train, y_train, X_test, y_test,
                beta_plus=beta, beta_minus=beta,
                filename=f"weights_beta_{beta}.txt"
            )

            all_stabilities.append(stabilities)
            all_Ea.append(Ea)
            all_Eg.append(Eg)

        # Graphique 1: Boxplot des stabilités
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.boxplot(all_stabilities, positions=beta_values)
        plt.xlabel('β (β+ = β-)')
        plt.ylabel('Stabilités sur test')
        plt.title('Distribution des stabilités en fonction de β')
        plt.grid(True, alpha=0.3)

        # Graphique 2: Moyenne des stabilités et erreurs
        plt.subplot(1, 2, 2)
        mean_stabilities = [np.mean(s) for s in all_stabilities]
        std_stabilities = [np.std(s) for s in all_stabilities]

        plt.errorbar(beta_values, mean_stabilities, yerr=std_stabilities,
                     marker='o', label='Stabilité moyenne', capsize=5)
        plt.plot(beta_values, all_Ea, 's-', label='Ea')
        plt.plot(beta_values, all_Eg, 'd-', label='Eg')

        plt.xlabel('β (β+ = β-)')
        plt.ylabel('Valeur')
        plt.title('Stabilités moyennes et erreurs')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('stabilities_vs_beta.png', dpi=150)
        plt.show()

        # Retourner les résultats
        results = {
            'beta_values': list(beta_values),
            'stabilities': all_stabilities,
            'Ea': all_Ea,
            'Eg': all_Eg
        }

        return results

    def test_with_two_temp(X_train, y_train, X_test, y_test):
        """
        Test avec deux températures différentes (rapport non unité).
        """
        # Exemple avec rapport de températures = 2
        print("\n" + "=" * 60)
        print("TEST AVEC RAPPORT β+/β- = 2")
        print("=" * 60)

        model, Ea, Eg, stabilities = run_experiment_two_temp(
            X_train, y_train, X_test, y_test,
            beta_plus=2.0, beta_minus=1.0,
            filename="weights_ratio_2.txt"
        )

        # Analyse des stabilités par classe
        if y_test is not None:
            y_test_bin = model._prepare_labels(y_test)
            stabilities_class1 = stabilities[y_test_bin == 1]
            stabilities_class2 = stabilities[y_test_bin == -1]

            print(f"\nStabilités - Classe +1: moy={np.mean(stabilities_class1):.3f}")
            print(f"Stabilités - Classe -1: moy={np.mean(stabilities_class2):.3f}")

        return model, Ea, Eg

    # ------------------------------------------------------
    # Exemple d'utilisation
    # ------------------------------------------------------

    if __name__ == "__main__":
        # Génération de données d'exemple
        np.random.seed(42)
        n_train, n_test, n_features = 100, 50, 10

        # Données d'entraînement
        X_train = np.random.randn(n_train, n_features)
        w_true = np.random.randn(n_features + 1)
        X_train_bias = np.hstack([X_train, np.ones((n_train, 1))])
        y_train = np.sign(X_train_bias @ w_true)

        # Données de test
        X_test = np.random.randn(n_test, n_features)
        X_test_bias = np.hstack([X_test, np.ones((n_test, 1))])
        y_test = np.sign(X_test_bias @ w_true)

        # 1. Test avec rapport β+/β- = 1 (variation de β)
        print("\n=== PARTIE d) : Graphique des stabilités vs β ===")
        results = plot_stabilities_vs_beta(X_train, y_train, X_test, y_test)

        # 2. Test avec deux températures différentes
        print("\n=== TEST AVEC DEUX TEMPÉRATURES DIFFÉRENTES ===")
        model, Ea, Eg = test_with_two_temp(X_train, y_train, X_test, y_test)
