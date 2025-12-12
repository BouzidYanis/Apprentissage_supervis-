# TP Perceptron – Apprentissage Supervisé

## Description
Ce projet implémente l'algorithme du perceptron en version batch et en ligne, dans le cadre d’un TP de M1 Intelligence Artificielle. Il permet de :
- Générer des données linéairement séparables avec un *perceptron professeur*
- Entraîner un *perceptron élève* sur ces données
- Comparer les performances (nombre d'itérations, recouvrement des poids) en fonction de la dimension, du nombre d'exemples et du taux d'apprentissage
- Visualiser les résultats sous forme de tableaux et de graphiques

---

## Structure du projet

### TP-Perceptron/

### ├── main.py : Point d'entrée principal – exécute les différentes questions du TP
### ├── Classs.py : Contient les classes Perceptron et GenerateLS
### ├── threads.py : Gestion du multithreading pour l'entraînement parallèle
### ├── requirements.txt :  Dépendances Python (optionnel)
### └── README.md : Ce fichier


## Classes principales
### 1. Perceptron (Classs.py)
Méthodes :

- train_with_batch(X, y) : version batch de l'apprentissage

- train_online(X, y) : version en ligne (stochastique)

- afficher_graphique(X, y, name) : affiche les points et la frontière de décision

Attributs :

- w : vecteur de poids après apprentissage

- epoch : nombre d'itérations effectuées

- errors_history : historique des erreurs par époque

### 2. GenerateLS (Classs.py)
Méthodes :

- get_ls() : génère un jeu de données linéairement séparable à l'aide d'un perceptron professeur

Attributs :

- perceptron_maitre : vecteur de poids du perceptron professeur

### 3. threads.py
Fonctions :

- thread_perceptron() : entraîne plusieurs perceptrons en parallèle et affiche les résultats

- thread_perceptron_v2() : version utilisée pour les tests systématiques (question 3)

- worker() : fonction exécutée par chaque thread

- afficher_results() : affichage tabulaire

- plot_perceptrons() : visualisation des perceptrons (maître et élèves)