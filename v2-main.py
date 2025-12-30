import numpy as np
from tools import load_breast_cancer_data, parse_sonar_file, data_split
from v2 import Minimerror, MinimerrorTwoTemp
from get_data import NParityDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler


def question_1():
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
# X_train, X_test, y_train, y_test = load_breast_cancer_data(test_size=0.2, random_state=42)


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

    y_pred_test = model.predict(X_test)
    Eg = np.sum(y_pred_test != y_test)

    print(f"Ea = {Ea}/{len(y_train)}")
    print(f"Eg = {Eg}/{len(y_test)}")

    pass


# parti_I()

def parti_II():
    def run_experiment_two_temp(X_train, y_train, X_test, y_test,
                                beta_plus=2.0, rapport_temperature=10,
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
        print(f"EXPÉRIENCE AVEC β+={beta_plus}, β-={beta_plus / rapport_temperature} ")
        print("=" * 60)

        # 1. Entraînement
        model = MinimerrorTwoTemp(
            beta0=beta_plus,
            rapport_temperature=rapport_temperature,
            learning_rate=0.02,
            init_method="hebb",
            normalize_weights=True
        )

        history = model.train(X_train, y_train, epochs=5000, verbose=True, beta_max=20)

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
        print(f"stabilities_test: {stabilities_test}")

        return model, Ea, Eg, stabilities_test

    def plot_stabilities_vs_beta(X_train, y_train, X_test, y_test,
                                 beta_values=range(1, 11)):
        """
        Graphique des stabilités en fonction de β (avec rapport fixé à 1).
        """
        all_stabilities = []
        all_Ea = []
        all_Eg = []

        rt = 5

        print("\n" + "=" * 60)
        print("ÉTUDE DES STABILITÉS EN FONCTION DE β")
        print("=" * 60)

        for beta in beta_values:
            print(f"\nβ = {beta}")

            # Entraînement avec beta_plus = beta_minus = beta
            model, Ea, Eg, stabilities = run_experiment_two_temp(
                X_train, y_train, X_test, y_test, 
                beta_plus=beta, rapport_temperature=rt,
                filename=f"weights_beta_{beta}.txt"
            )

            all_stabilities.append(stabilities)
            all_Ea.append(Ea)
            all_Eg.append(Eg)

        # Graphique 1: Boxplot des stabilités
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.boxplot(all_stabilities, positions=beta_values)
        plt.xlabel('β')
        plt.ylabel('Stabilités sur test')
        plt.title(f'Distribution des stabilités en fonction de β avec rapport={rt}')
        plt.grid(True, alpha=0.3)

        # Graphique 2: Moyenne des stabilités et erreurs
        plt.subplot(1, 2, 2)
        mean_stabilities = [np.mean(s) for s in all_stabilities]
        std_stabilities = [np.std(s) for s in all_stabilities]

        plt.errorbar(beta_values, mean_stabilities, yerr=std_stabilities,
                     marker='o', label='Stabilité moyenne', capsize=5)
        plt.plot(beta_values, all_Ea, 's-', label='Ea')
        plt.plot(beta_values, all_Eg, 'd-', label='Eg')

        plt.xlabel('β ')
        plt.ylabel('Valeur')
        plt.title(f'Stabilités moyennes et erreurs en fonction de β et rapport={rt}')
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
            beta_plus=2.0, rapport_temperature=10,
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
    def linear_separable_gaussians(n=200, d=2, margin=2.0, seed=42):
        np.random.seed(seed)

        X_pos = np.random.randn(n//2, d) + margin
        X_neg = np.random.randn(n//2, d) - margin

        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(n//2), -np.ones(n//2)])

        return X, y

    if __name__ == "__main__":
        # Génération de données d'exemple
        
        x_train, x_test, y_train, y_test = load_breast_cancer_data(test_size=0.5, random_state=42)
        model = MinimerrorTwoTemp(
            beta0=0.1,                    # β initial petit
            rapport_temperature=1,       # ρ = β+/β- = 6
            learning_rate=0.02,           # Taux d'apprentissage modéré
            init_method="hebb",           # Initialisation Hebb
            hebb_noise=1e-3,              # Léger bruit pour Hebb
            normalize_weights=True,        # Normalisation des poids
            scale_inputs=False
        )

        history = model.train(
            x_train, y_train, 
            epochs=5000,
            beta_max=100,           # Température maximale
        )


        # 2. Calcul des erreurs
        y_pred = model.predict(x_train)
        Ea = np.sum(y_pred != y_train)
        print(f"\nEa (erreur apprentissage) = {Ea:.4f}")

# parti_II()

def test_with_perceptron_sklearn():
    X_train, X_test,y_train, y_test = load_breast_cancer_data(0.3, 42)
    perceptron = Perceptron(max_iter=10000, eta0=0.2)
    perceptron.fit(X_train, y_train)
    y_predit = perceptron.predict(X_train)
    ea = np.sum(y_predit != y_train)
    print(f"Erreurs sur les données d'entraînement : {ea}")

def test_digits_slow_convergence():
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # --- Préparation des données (identique) ---
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    mask = y < 2 
    X = X[mask]
    y = y[mask]
    y_formatted = np.where(y == 0, -1, 1)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    print(f"Dataset Digits (0 vs 1) : {X.shape[0]} exemples")
    print("Démarrage du test de convergence LENTE...")

    # --- CONFIGURATION "TORTUE" ---
    model = MinimerrorTwoTemp(
        beta0=2.0,                 # On commence déjà "froid" (gradients plus faibles loin de la frontière)
        rapport_temperature=1.0,
        learning_rate=0.001,      # Pas d'apprentissage minuscule (100x plus petit que la normale)
        init_method="random",      # ON CASSE L'INTELLIGENCE DE HEBB : Départ au hasard
        hebb_noise=0.0,            # Inutile en random
        normalize_weights=True,
        scale_inputs=False,        
        momentum=0.0               # Pas d'élan pour aider à accélérer
    )

    # --- Entraînement ---
    # On met beaucoup d'epochs car ça va être long
    history = model.train(
        X_scaled, y_formatted, 
        epochs=10000,              # Il lui faudra peut-être 2000 à 5000 époques
        early_stopping=True, 
        verbose=True, 
        beta_max=50
    )

    # --- Vérification ---
    y_pred = model.predict(X_scaled)
    Ea = np.sum(y_pred != y_formatted)
    print(f"Résultat Final : Ea = {Ea} / {len(y_formatted)}")

# test_digits_slow_convergence()

x_train, y_train = None, None
x_test, y_test = None, None
x_all, y_all = None, None

file_path = r"./data"
def question_1():
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
    y_train['class'] = y_train['class'].map({'M': 1, 'R': -1})
    y_test['class'] = y_test['class'].map({'M': 1, 'R': -1})
    y_all['class'] = y_all['class'].map({'M': 1, 'R': -1})

    x_train = x_train.to_numpy()
    y_train = y_train['class'].to_numpy() if hasattr(y_all, 'columns') and 'class' in y_all.columns else y_all.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test['class'].to_numpy() if hasattr(y_test, 'columns') and 'class' in y_test.columns else y_test.to_numpy()
    x_all = x_all.to_numpy()
    y_all = y_all['class'].to_numpy() if hasattr(y_all, 'columns') and 'class' in y_all.columns else y_all.to_numpy()

def test_with_sonar():
    question_1() # Prépare x_train, y_train
    
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    
    model = MinimerrorTwoTemp(
        beta0=0.2,             # Température de départ
        rapport_temperature=15, # Un rapport modéré est bon ici
        learning_rate=0.02,    # Pas trop élevé pour ne pas rater le minimum
        init_method="hebb",    # GARDER HEBB pour le Sonar
        hebb_noise=0.01,
        normalize_weights=True,
        scale_inputs=False,    # Déjà fait manuellement avec sc
    )
    
    # MODIFICATIONS ICI : 
    # - Réduire beta_max : 50 ou 100 suffisent largement pour le Sonar
    # - Réduire les epochs : 60 000 est excessif et risque de faire exploser le gradient
    history = model.train(x_train_scaled, y_train, 
                         epochs=3000, 
                         early_stopping=True, 
                         verbose=True, 
                         beta_max=120,b=0.995) # <--- IMPORTANT
    
    y_predict = model.predict(x_train_scaled)
    print(f"\nEa (erreur apprentissage) = {np.sum(y_predict != y_train)} / {len(y_train)}")
    y_predict_test = model.predict(sc.transform(x_test))
    print(f"Eg (erreur généralisation) = {np.sum(y_predict_test != y_test)} / {len(y_test)}")
test_with_sonar()
def grid_search_sonar():
    question_1() # Charge x_train, y_train
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    
    # Espace de recherche
    raccourcis = {
        'rapports': [4, 6, 8, 10],
        'beta_maxs': [50, 80, 100, 150],
        'lrs': [0.01, 0.02]
    }
    
    best_ea = len(y_train)
    best_params = {}

    print(f"{'Rapport':<10} | {'Beta_max':<10} | {'LR':<10} | {'Ea':<5}")
    print("-" * 45)

    for r in raccourcis['rapports']:
        for b_max in raccourcis['beta_maxs']:
            for lr in raccourcis['lrs']:
                model = MinimerrorTwoTemp(
                    beta0=0.1,
                    rapport_temperature=r,
                    learning_rate=lr,
                    init_method="hebb",
                    hebb_noise=0.01,
                    normalize_weights=True,
                    scale_inputs=False
                )
                
                # On utilise un grand nombre d'époques pour garantir la convergence
                model.train(x_train_scaled, y_train, epochs=2000, 
                            beta_max=b_max, verbose=False, early_stopping=True, b=0.99)
                
                y_pred = model.predict(x_train_scaled)
                ea = np.sum(y_pred != y_train)
                
                print(f"{r:<10} | {b_max:<10} | {lr:<10} | {ea:<5}")
                
                if ea < best_ea:
                    best_ea = ea
                    best_params = {'rapport': r, 'beta_max': b_max, 'lr': lr}
                
                if ea == 0:
                    print("!!! ZÉRO ERREUR ATTEINT !!!")
                    return best_params

    return best_params

# Lancer la recherche
# params = grid_search_sonar()

def find_zero_ea_sonar():
    question_1()
    sc = StandardScaler()
    X_scaled = sc.fit_transform(x_train)
    
    # On resserre la grille autour de vos paramètres qui donnent 2 erreurs
    grid = {
        'rapports': [8, 10, 12, 15],      # On monte un peu le rapport pour durcir la marge
        'beta_maxs': [80, 100, 120, 150], # Exploration autour de 100
        'lrs': [0.01, 0.02]               # Test de deux vitesses d'apprentissage
    }
    
    print(f"{'Rapport':<8} | {'Beta_max':<8} | {'LR':<6} | {'Ea':<5}")
    print("-" * 35)

    for r in grid['rapports']:
        for b_max in grid['beta_maxs']:
            for lr in grid['lrs']:
                model = MinimerrorTwoTemp(
                    beta0=0.1,
                    rapport_temperature=r,
                    learning_rate=lr,
                    init_method="hebb",
                    hebb_noise=0.01,
                    normalize_weights=True,
                    scale_inputs=False
                )
                
                # On utilise 3000 époques pour être certain de ne pas s'arrêter trop tôt
                # On baisse le LR très doucement (0.999) pour garder de la précision
                model.train(X_scaled, y_train, epochs=3000, beta_max=b_max, 
                            verbose=False, early_stopping=True, b=0.995)
                
                y_pred = model.predict(X_scaled)
                ea = np.sum(y_pred != y_train)
                
                print(f"{r:<8} | {b_max:<8} | {lr:<6} | {ea:<5}")
                
                if ea == 0:
                    print("\n>>> SUCCÈS : Ea = 0 trouvé !")
                    return {'rapport': r, 'beta_max': b_max, 'lr': lr}
                    
    return None

best_params = find_zero_ea_sonar()
print("\nMeilleurs paramètres trouvés :", best_params)
# print("\nMeilleurs paramètres trouvés :", params)
def optimize_sonar_zero_error():
    # 1. Préparation des données
    question_1()
    global x_train, y_train
    
    # Conversion explicite des labels pour être sûr (0/1 -> -1/1)
    # Minimerror attend du -1/1 en interne, mais on veut être sûr de l'entrée
    
    # Standardisation OBLIGATOIRE pour le Sonar
    sc = StandardScaler()
    X_scaled = sc.fit_transform(x_train)
    
    print(f"Dataset : {X_scaled.shape[0]} exemples, {X_scaled.shape[1]} features")
    print("-" * 60)

    # 2. Paramètres à tester
    # On va tester plusieurs "violences" de bruit et de température
    best_overall_Ea = float('inf')
    best_config = None
    best_weights = None

    # On fait plusieurs essais (restarts) car le résultat dépend du bruit aléatoire
    n_restarts = 10 
    
    for i in range(n_restarts):
        # Paramètres un peu aléatoires pour explorer
        current_noise = 0.05 + (i * 0.02)  # Augmente le bruit à chaque essai
        
        print(f"Tentative {i+1}/{n_restarts} (Bruit Hebb: {current_noise:.2f})...")
        
        model = MinimerrorTwoTemp(
            beta0=0.5,                 # Température de départ modérée
            rapport_temperature=2.0,   # Asymétrie : on serre plus une classe que l'autre
            learning_rate=0.02,        # Pas trop petit pour bouger, pas trop grand pour converger
            init_method="hebb",
            hebb_noise=current_noise,  # CRUCIAL : Casser la symétrie
            normalize_weights=True,
            scale_inputs=False,        # Déjà fait
            momentum=0.6,              # Inertie pour franchir les petits obstacles
            min_lr_ratio=0.01
        )

        # Entraînement
        model.train(
            X_scaled, y_train, 
            epochs=2000, 
            early_stopping=True, 
            verbose=False, 
            beta_max=200  # Pas besoin de 10000, 200 suffit pour saturer tanh
        )

        # Vérification
        y_pred = model.predict(X_scaled)
        Ea = np.sum(y_pred != y_train)
        
        print(f"   -> Ea = {Ea}")
        
        if Ea < best_overall_Ea:
            best_overall_Ea = Ea
            best_config = f"Noise={current_noise:.2f}"
            best_weights = model.w.copy()
            
        if Ea == 0:
            print("\n>>> VICTOIRE ! Séparation parfaite trouvée (Ea=0).")
            break

    print("=" * 60)
    print(f"MEILLEUR RÉSULTAT : {best_overall_Ea} erreurs")
    print(f"Config gagnante : {best_config}")
    
    if best_overall_Ea > 0:
        print("Note : Si Ea reste > 0 (ex: 1 ou 2), le dataset n'est probablement pas linéairement séparable.")

# Lancer l'optimisation
# optimize_sonar_zero_error()

def test_with_perceptron_sklearn_sonar():
    question_1()
    global x_train, y_train
    global x_test, y_test
    perceptron = Perceptron(max_iter=10000, eta0=0.5)
    perceptron.fit(x_train, y_train)
    y_predit = perceptron.predict(x_train)
    ea = np.sum(y_predit != y_train)
    print(f"Erreurs sur les données d'entraînement : {ea}")
# test_with_perceptron_sklearn_sonar()

def test_Nparity():
    n = 10
    np.random.seed(0)
    dataset = NParityDataset(n)
    X = dataset.X
    y = dataset.y
    model = MinimerrorTwoTemp(
        beta0=0.2,
        rapport_temperature=6,       # Recommandation thèse: 6
        learning_rate=0.02,          # Recommandation thèse: 0.02
        init_method="random",        # CRITIQUE pour la Parité [cite: 1000]
        normalize_weights=True,
        scale_inputs=True,
        min_lr_ratio=0.01
    )
    history = model.train(X, y, epochs=300, beta_max=4000, early_stopping=False,verbose=True)
    y_pred = model.predict(X)
    Ea = int(np.sum(y_pred != y))
    print(f"Ea (erreur apprentissage) = {Ea} / {len(y)}")

test_Nparity()


def test_Nparity_perceptron():
    np.random.seed(0)
    n = 10
    n_parity = NParityDataset(n)
    X, y = n_parity.generate()
    perceptron = Perceptron(max_iter=10000, eta0=0.001)
    perceptron.fit(X, y)
    y_predit = perceptron.predict(X)
    ea = np.sum(y_predit != y)
    print(f"Erreurs sur les données d'entraînement : {ea} / {len(y)}")
# test_Nparity_perceptron()
