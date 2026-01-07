from Monoplan import Monoplan
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import NParityDataset

# Exemple d'utilisation
if __name__ == "__main__":
    # Générer des données d'exemple (problème XOR)
    np.random.seed(0)
    n = 5
    dataset = NParityDataset(n)
    X, y = dataset.X, dataset.y
    
    # Créer et entraîner le modèle
    model = Monoplan(N=n, P=2^n, H_max=200)
    model.train(X, y)
    
    # Faire des prédictions
    predictions = model.predict(X)
    print(f"\nPrédictions: {predictions}")
    print(f"Étiquettes réelles: {y}")
    print(f"Nombre d'erreurs: {np.sum(predictions != y)}")
