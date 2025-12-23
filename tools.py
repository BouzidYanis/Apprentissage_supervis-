import re
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def parse_sonar_file(filename, path):
    """
    Parse les fichiers sonar.names et sonar.mines :
    '*' => train ; absence de '*' => test

    Retourne 4 DataFrames:
      - train_df (60 colonnes)
      - test_df  (60 colonnes)
      - y_train_df (1 colonne 'class')
      - y_test_df  (1 colonne 'class')
    """
    p = Path(path) / filename
    if not p.exists():
        raise FileNotFoundError(f"Fichier introuvable: {p}")

    with open(p, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex généreuse : accepte IDs de type lettres+chiffres
    pattern = r'^\s*(\*?)\s*([A-Za-z]+\d+)\s*:\s*\{([^}]*)\}'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    if len(matches) == 0:
        raise ValueError(
            "Aucune entrée trouvée. Vérifie que les lignes ressemblent à '*CM123: {0.1, ...}' "
            "et que les accolades '{ }' et les IDs sont présents."
        )

    train_data = {}
    test_data = {}

    for star, sample_id, values_str in matches:
        # Extraire nombres (supporte 1, 0.5, -0.03)
        nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', values_str)
        values = [float(v) for v in nums]

        if len(values) != 60:
            raise ValueError(f"{sample_id} a {len(values)} dimensions (attendu 60). Vérifie la ligne.")

        if star == '*':
            train_data[sample_id] = values
        else:
            test_data[sample_id] = values

    # Avant de renommer: vérifier qu’on a trouvé au moins une entrée
    if len(train_data) == 0 and len(test_data) == 0:
        raise ValueError("Aucune donnée parsée. Le fichier ne contient pas de lignes conformes au format attendu.")

    # Construire DataFrames
    cols = [f'Attribute{i}' for i in range(1, 61)]
    train_df = pd.DataFrame.from_dict(train_data, orient='index')
    test_df = pd.DataFrame.from_dict(test_data, orient='index')

    # Renommer les colonnes (seulement si DataFrame non vide)
    if train_df.shape[1] > 0:
        train_df.columns = cols
    if test_df.shape[1] > 0:
        test_df.columns = cols

    train_df.index.name = 'Sample_ID'
    test_df.index.name = 'Sample_ID'

    # Déterminer le label depuis le nom du fichier
    fn_lower = filename.lower()
    if 'mine' in fn_lower:
        label = 'M'
    elif 'rock' in fn_lower:
        label = 'R'
    else:
        raise ValueError("Impossible d'inférer le label depuis le nom du fichier (attendu 'mines' ou 'rocks').")

    y_train_df = pd.DataFrame({'class': [label] * len(train_df)}, index=train_df.index)
    y_test_df = pd.DataFrame({'class': [label] * len(test_df)}, index=test_df.index)

    return train_df, test_df, y_train_df, y_test_df


def load_breast_cancer_data(test_size, random_state):
    """
    Charge et prépare les données Breast Cancer Wisconsin.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Convertir {0,1} à {-1,1}
    y = 2 * y - 1

    # Normaliser
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = data_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def data_split(X, y, test_size=0.2, random_state=None, shuffle=True, stratify=None):

    # Convertir en numpy arrays si nécessaire
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = len(X)

    # Déterminer le nombre d'échantillons de test
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    # Vérifier que test_size est valide
    if n_test == 0 or n_train == 0:
        raise ValueError(f"test_size={test_size} donne n_test={n_test} ou n_train={n_train}")

    # Initialiser le générateur aléatoire
    if random_state is not None:
        np.random.seed(random_state)

    # Créer les indices
    indices = np.arange(n_samples)

    # Split stratifié si demandé
    if stratify is not None:
        stratify = np.asarray(stratify)
        unique_classes, class_counts = np.unique(stratify, return_counts=True)

        train_indices = []
        test_indices = []

        for cls, count in zip(unique_classes, class_counts):
            # Indices pour cette classe
            cls_indices = indices[stratify == cls]

            # Nombre de test pour cette classe (proportionnel)
            n_test_cls = max(1, int(count * test_size))

            # Mélanger les indices de cette classe
            if shuffle:
                np.random.shuffle(cls_indices)

            # Split pour cette classe
            cls_test_indices = cls_indices[:n_test_cls]
            cls_train_indices = cls_indices[n_test_cls:]

            test_indices.extend(cls_test_indices)
            train_indices.extend(cls_train_indices)

        # Convertir en arrays
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Mélanger final si demandé (mais en gardant la stratification)
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

    else:
        # Split simple (non stratifié)
        if shuffle:
            np.random.shuffle(indices)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    # Séparer les données
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test