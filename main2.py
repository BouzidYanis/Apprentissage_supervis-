from Classs import Perceptron, GenerateLS
import numpy as np
import pandas as pd
from tools import parse_sonar_file

"""
    Variables Globales
"""
file_path = r"./data"
x_train, y_train = None, None
x_test, y_test = None, None
x_all, y_all = None, None


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


""""
    2 Apprentissage sur « train ». Utiliser l’algorithme du perceptron (justifier le choix
    version batch vs online selon votre TP1) pour apprendre l’ensemble « train », puis tester
    sur l’ensemble de « test ».
    a) Calculer les erreurs d’apprentissage Ea et de généralisation Eg ;
    b) Afficher les N+1 poids W du perceptron ;
    c) Calculer les stabilités des P exemples de « test » selon la formule de gamma (distance a l’hyperplan séparateur avec les poids normés)
    d) Graphique des stabilités
"""


def question_2():

    # Il faut transformer les données catégorielles  M, R en 1, 0
    y_train['class'] = y_train['class'].map({'M': 1, 'R': 0})
    y_test['class'] = y_test['class'].map({'M': 1, 'R': 0})

    perceptron = Perceptron(eta=0.5, max_epoch=1000)
    perceptron.train_online(x_train.to_numpy(), y_train.to_numpy())

    # Prédictions
    _ = perceptron.predict(x_test=x_test.to_numpy())

    # Erreur d'apprentissage
    ea = perceptron.get_ea()

    # Erreur de généralisation
    eg = perceptron.get_eg(x_test.to_numpy(), y_test.to_numpy())

    print(f"""L'erreur d'apprentissage Ea est : {ea} \n
              L'erreur de généralisation Eg : {eg} """)

    print(f"Les n+1 poids w du perceptron : {perceptron.w}")

    print("Calcul des stabilités des P exemples de « test » selon la formule de gamma ")

    stabilities = perceptron.compute_stability(x_train.to_numpy(), y_train.to_numpy())

    print(f"Matrice de stabilité : {stabilities}")

    print(f"Stabilité moyenne : {np.mean(stabilities):.4f}")

    print("Graphique simple des stabilités")
    perceptron.plot_stability_geometric(x_train.to_numpy(), y_train.to_numpy(),
                                        title="Stabilités géométriques")

    # Pour voir juste la représentation 2D de base
    """ perceptron._plot_2d_geometric(x_train.to_numpy(), y_train.to_numpy(),
                                  perceptron.compute_stability(x_train.to_numpy(), y_train.to_numpy()),
                                  "Représentation 2D")"""

    # perceptron.plot_stability_analysis(x_train.to_numpy(), y_train, X_test, y_test)


question_1()
question_2()









