from get_data import GenerateLS
from v2 import MinimerrorTwoTemp
import numpy as np
import matplotlib.pyplot as plt
from tools import data_split
import pandas as pd

def run_experiment_two_temp(X_train, y_train, X_test, y_test,
                                beta_plus=2.0, rapport_temperature=10):
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
    history = model.train(X_train, y_train, epochs=5000, verbose=False, beta_max=20)
    # 2. Calcul des erreurs
    Ea, Eg = model.compute_errors(X_train, y_train, X_test, y_test)
    print(f"\nEa (erreur apprentissage) = {Ea:.4f}")
    print(f"Eg (erreur généralisation) = {Eg:.4f}")
    # 4. Calcul des stabilités sur test
    stabilities_test = model.compute_test_stabilities(X_test)
    print(f"Stabilités test: moy={np.mean(stabilities_test):.3f}, "
          f"std={np.std(stabilities_test):.3f}")
    print(f"stabilities_test: {stabilities_test}")
    return model, Ea, Eg, stabilities_test

# def plot_stabilities_vs_beta(X_train, y_train, X_test, y_test,beta_values=range(1, 11)):
#     """
#     Graphique des stabilités en fonction de β (avec rapport fixé à 1).
#     """
#     all_stabilities = []
#     all_Ea = []
#     all_Eg = []
#     rt = 5
#     print("\n" + "=" * 60)
#     print("ÉTUDE DES STABILITÉS EN FONCTION DE β")
#     print("=" * 60)
#     for beta in beta_values:
#         print(f"\nβ = {beta}")
#         # Entraînement avec beta_plus = beta_minus = beta
#         model, Ea, Eg, stabilities = run_experiment_two_temp(
#             X_train, y_train, X_test, y_test, 
#             beta_plus=beta, rapport_temperature=rt,
#         )
#         all_stabilities.append(stabilities)
#         all_Ea.append(Ea)
#         all_Eg.append(Eg)
#     # Graphique 1: Boxplot des stabilités
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.boxplot(all_stabilities, positions=beta_values)
#     plt.xlabel('β')
#     plt.ylabel('Stabilités sur test')
#     plt.title(f'Distribution des stabilités en fonction de β avec rapport={rt}')
#     plt.grid(True, alpha=0.3)
#     # Graphique 2: Moyenne des stabilités et erreurs
#     plt.subplot(1, 2, 2)
#     mean_stabilities = [np.mean(s) for s in all_stabilities]
#     std_stabilities = [np.std(s) for s in all_stabilities]
#     plt.errorbar(beta_values, mean_stabilities, yerr=std_stabilities,
#                  marker='o', label='Stabilité moyenne', capsize=5)
#     plt.plot(beta_values, all_Ea, 's-', label='Ea')
#     plt.plot(beta_values, all_Eg, 'd-', label='Eg')
#     plt.xlabel('β ')
#     plt.ylabel('Valeur')
#     plt.title(f'Stabilités moyennes et erreurs en fonction de β et rapport={rt}')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('stabilities_vs_beta.png', dpi=150)
#     plt.show()
#     # Retourner les résultats
#     results = {
#         'beta_values': list(beta_values),
#         'stabilities': all_stabilities,
#         'Ea': all_Ea,
#         'Eg': all_Eg
#     }
#     return results

def plot_stability_vs_beta(X, y, beta_values=range(1, 11)):
    """
    Entraîne des modèles pour différentes valeurs finales de beta (ou capture l'état)
    et affiche la distribution des stabilités sous forme de Boxplot.

    Args:
        X: Données d'entrée
        y: Labels
        beta_values: Liste des valeurs de beta à tester (ex: 1, 2, ... 10)
    """
    
    results = []
    
    # Préparation des données pour l'affichage
    # On convertit y en -1/1 pour être sûr des calculs de stabilité
    
    # On standardise une fois pour toutes si nécessaire
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Calcul des stabilités en cours pour beta =", end=" ")
    
    for b in beta_values:
        print(f"{b}...", end=" ", flush=True)
        
        # On configure le modèle pour qu'il finisse son recuit exactement à beta_max = b
        # On peut réduire les époques si b est petit, mais gardons constant pour comparer
        model = MinimerrorTwoTemp(
            beta0=0.1,                 # Départ chaud
            rapport_temperature=2.0,   # Exemple standard
            learning_rate=0.01,
            init_method="hebb",
            hebb_noise=0.01,
            normalize_weights=True,
            scale_inputs=False         # Déjà fait
        )
        
        # Entraînement jusqu'à atteindre la température b
        model.train(
            X_scaled, y, 
            epochs=1000,               # Suffisant pour stabiliser
            beta_max=b,                # C'est ici que ça se joue
            early_stopping=False,      # On veut voir l'état à la fin, même si Ea=0
            verbose=False
        )
        
        # Récupération des stabilités finales
        stabs = model.compute_stability(X_scaled, y)
        
        # Stockage pour le DataFrame
        for s in stabs:
            results.append({'Beta': b, 'Stabilité': s})
            
    print("Terminé.")
    
    # Création d'un DataFrame pour faciliter le plot avec Seaborn/Pandas
    df = pd.DataFrame(results)

    # --- Graphique ---
    plt.figure(figsize=(12, 6))
    
    # Ligne zéro (Frontière de décision)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label="Frontière (Hyperplan)")
    
    # Boxplot
    # Si Seaborn est disponible (plus joli)
    try:
        import seaborn as sns
        sns.boxplot(x='Beta', y='Stabilité', data=df, palette="viridis", width=0.5)
        sns.stripplot(x='Beta', y='Stabilité', data=df, color='black', alpha=0.3, size=2) # Ajoute les points bruts
    except ImportError:
        # Version Matplotlib standard
        data_to_plot = [df[df['Beta'] == b]['Stabilité'].values for b in beta_values]
        plt.boxplot(data_to_plot, labels=beta_values, patch_artist=True)

    plt.title("Distribution des Stabilités en fonction de β (Inverse Température)")
    plt.xlabel("Valeur de β (Dureté de la fonction coût)")
    plt.ylabel("Stabilité γ (Distance signée normalisée)")
    plt.grid(True, alpha=0.3)
    
    # Interprétation visuelle
    plt.text(0.5, df['Stabilité'].min(), "Zone d'erreur (γ < 0)", color='red', ha='center')
    plt.text(0.5, df['Stabilité'].max(), "Zone de confiance (γ > 0)", color='green', ha='center')
    
    plt.show()

def parti_II():
     

   

    if __name__ == "__main__":
        # Génération de données d'exemple
        beta_initial = 2.0
        rapport_temperature = 2.0
        np.random.seed(0)
        ls = GenerateLS(n=4, p=100)
        X, y = ls.get_ls()
        # Split train/test
        x_train, x_test, y_train, y_test = data_split(X, y, test_size=0.2, random_state=42)
        print("=" * 60)
        print(f"EXPÉRIENCE AVEC β+={beta_initial}, β-={beta_initial / rapport_temperature} ")
        print("=" * 60)

        # 1. Entraînement
        model = MinimerrorTwoTemp(
            beta0=beta_initial,                    # β initial petit
            rapport_temperature=rapport_temperature,       # ρ = β+/β- = 6
            learning_rate=0.001,           # Taux d'apprentissage modéré
            init_method="hebb",           # Initialisation Hebb
            hebb_noise=1e-3,              # Léger bruit pour Hebb
            normalize_weights=True,        # Normalisation des poids
            scale_inputs=False
        )

        history = model.train(
            x_train, y_train, 
            epochs=800,
            beta_max=1000,           # Température maximale
            b= 1
        )
         # 2. Calcul des erreurs
        y_predict = model.predict(x_train)
        Ea = int(np.sum(y_predict != y_train))
        print(f"\nEa (erreur apprentissage) = {Ea:.4f}")

        # 3. Sauvegarde des poids
        model.save_weights("weights.txt")

        stabilites_test = model.compute_stability(x_test, y_test)
        print("Stabilités test:", stabilites_test)
        print("Étude des stabilités en fonction de β avec rapport fixé à 2")
        # plot_stability_vs_beta(x_test, y_test)


parti_II()