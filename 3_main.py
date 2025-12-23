from MinimError import NParityDataset, Minimerror
from MinimError import plot_minimerror_pca, plot_cost_and_derivative
import numpy as np
import pandas as pd
from tools import parse_sonar_file
import matplotlib.pyplot as plt


def init_1():
    np.random.seed(0)
    n = 6  # 5
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

    # Numpy arrays
    Xn = x_train.to_numpy().astype(float)
    yn = y_train["class"].to_numpy().astype(float)

    # Entraînement perceptron sur l'ensemble L
    model = Minimerror(
        init_method="hebb",
        normalize_weights=True,
        scale_inputs=False,
        hebb_noise=1e-2,
        learning_rate=0.0001,
        T=2  # 150.0,
    )

    history = model.train(
        Xn, yn,
        epochs=500000,  # 10000
        T_final=1e-6 # T = 0.010

    )

    best_epoch = np.argmin(model.history['error'])
    final_error = model.history['error'][best_epoch]

    print("\nbest_epoch",  best_epoch + 1)
    print("\nRésultat final")
    print(f"Erreurs = {int(final_error * yn.shape[0])} / {yn.shape[0]}")

    print("\n Stabilité moyenne : ", np.mean(model.history["stabilities"][best_epoch]))
    plot_minimerror_pca(model=model, X=Xn, y=yn, title="Minimerror – PCA + hyperplan")

    plot_cost_and_derivative(model=model)
    model.save_weights("minimerror_sonar_weights.csv")




# ======================================================================
# Test principal
# ======================================================================

# init_1()
init_2()
init_3()

