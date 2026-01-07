import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from MinimError import Minimerror, MinimerrorTwoTemp

class Monoplan:
    def __init__(self, N, P, H_max=10, E_max=0):
        """
        Algorithme Monoplan pour l'apprentissage incrémental
        
        Args:
            N: Nombre d'entrées
            P: Nombre d'exemples d'apprentissage
            H_max: Nombre maximum de couches cachées (défaut: 10)
            E_max: Nombre maximum d'erreurs d'apprentissage (défaut: 0)
        """
        self.N = N
        self.P = P
        self.H_max = H_max
        self.E_max = E_max
        self.hidden_layers = []
        self.output_layer = None
        
    def train(self, X_train, y_train, verbose=True):
        """
        Entraîne le réseau avec l'algorithme Monoplan
        
        Args:
            X_train: Ensemble d'apprentissage (P x N)
            y_train: Étiquettes des exemples (P,)
            verbose: Afficher les détails (défaut: True)
        """
        h = 0  # Nombre de couches cachées
        tau = y_train.copy()
        
        # Boucle principale
        while True:
            # Boucle interne
            while True:
                # Construire la couche cachée
                h += 1
                if verbose:
                    print(f"\n=== Construction de la couche cachée {h} ===")
                
                # Préparer les entrées: X_train + sorties des couches cachées précédentes
                if h == 1:
                    X_input = X_train
                else:
                    # Concaténer X_train avec les sorties de toutes les couches précédentes
                    hidden_outputs = self._get_all_hidden_outputs(X_train)
                    X_input = np.column_stack([X_train, hidden_outputs])
                
                # Ajouter une colonne de biais
                X_with_bias = np.column_stack([np.ones(X_input.shape[0]), X_input])
                
                # Vérifier s'il y a plus d'une classe dans tau
                unique_targets = np.unique(tau)
                if len(unique_targets) == 1:
                    if verbose:
                        print(f"Toutes les cibles sont identiques ({unique_targets[0]})")
                    hidden_unit = TrivialClassifier(unique_targets[0])
                else:
                    # Entraîner avec Minimerror
                    hidden_unit = Minimerror(
                        T=10,
                        learning_rate=0.1,
                        init_method='hebb',
                        hebb_noise=0.01,
                        normalize_weights=True,
                        scale_inputs=True,
                        momentum=0.5,
                        min_lr_ratio=0.0001
                    )
                    hidden_unit.train(
                        X_with_bias, tau,
                        epochs=1000,
                        anneal=True,
                        T_final=0.01,
                        gradient_method='all',
                        early_stopping=True,
                        verbose=False
                    )
                
                sigma = hidden_unit.predict(X_with_bias)
                sigma = np.sign(sigma)
                self.hidden_layers.append(hidden_unit)
                print(sigma)
                
                # Mettre les nouveaux objectifs à apprendre
                tau_next = tau * sigma
                
                # Vérifier si L est linéairement séparable
                separable = np.all(tau_next == 1)
                
                if separable:
                    if verbose:
                        print("L est linéairement séparable!")
                    break
                
                # Compter le nombre d'erreurs d'apprentissage
                e_h = np.sum(1 - tau_next) / 2
                if verbose:
                    print(f"Nombre d'erreurs e_h = {e_h}")
                
                if e_h == 0:
                    break
                
                # Limiter le nombre de couches internes
                if h >= 10:
                    if verbose:
                        print(f"Limite de couches internes atteinte (h={h})")
                    break
                
                tau = tau_next
            
            # Apprendre la sortie
            if verbose:
                print(f"\n=== Apprentissage de la couche de sortie ===")
            
            X_hidden = self._get_all_hidden_outputs(X_train)
            X_hidden_with_bias = np.column_stack([np.ones(X_hidden.shape[0]), 
                                                   X_hidden])
            
            unique_outputs = np.unique(y_train)
            if len(unique_outputs) == 1:
                self.output_layer = TrivialClassifier(unique_outputs[0])
            else:
                self.output_layer = Minimerror(
                    T=10,
                    learning_rate=0.1,
                    init_method='hebb',
                    hebb_noise=0.01,
                    normalize_weights=True,
                    scale_inputs=True,
                    momentum=0.5,
                    min_lr_ratio=0.0001
                )
                self.output_layer.train(
                    X_hidden_with_bias, y_train,
                    epochs=1000,
                    anneal=True,
                    T_final=0.01,
                    gradient_method='all',
                    early_stopping=True,
                    verbose=False
                )
            
            zeta = self.output_layer.predict(X_hidden_with_bias)
            tau = y_train * zeta
            e_zeta = np.sum(1 - tau) / 2
            
            if verbose:
                print(f"Nombre d'erreurs de sortie e_ζ = {e_zeta}")
            
            # Vérifier la condition d'arrêt
            if h <= self.H_max and e_zeta > self.E_max:
                if verbose:
                    print(f"h={h} ≤ H_max={self.H_max} ET e_ζ={e_zeta} > E_max={self.E_max}")
                    print("Continuer l'apprentissage...\n")
            else:
                if verbose:
                    print(f"\nCondition d'arrêt atteinte!")
                    print(f"h={h}, H_max={self.H_max}, e_ζ={e_zeta}, E_max={self.E_max}")
                break
    
    def _get_all_hidden_outputs(self, X):
        """
        Obtenir les sorties de toutes les couches cachées
        """
        if len(self.hidden_layers) == 0:
            return np.array([]).reshape(X.shape[0], 0)
        
        all_outputs = []
        
        for i, layer in enumerate(self.hidden_layers):
            if i == 0:
                current_input = X
            else:
                previous_outputs = np.hstack(all_outputs)
                current_input = np.column_stack([X, previous_outputs])
            
            X_with_bias = np.column_stack([np.ones(current_input.shape[0]), 
                                           current_input])
            layer_output = layer.predict(X_with_bias)
            all_outputs.append(layer_output.reshape(-1, 1))
        
        return np.hstack(all_outputs)
    
    def predict(self, X):
        """Prédire les étiquettes pour de nouveaux exemples"""
        if not self.output_layer:
            raise ValueError("Le modèle n'a pas encore été entraîné!")
        
        X_hidden = self._get_all_hidden_outputs(X)
        X_hidden_with_bias = np.column_stack([np.ones(X_hidden.shape[0]), 
                                               X_hidden])
        predictions_bipolar = self.output_layer.predict(X_hidden_with_bias)
        return predictions_bipolar
        

class TrivialClassifier:
    """Classificateur trivial qui prédit toujours la même classe"""
    def __init__(self, label):
        self.label = label
    
    def predict(self, X):
        return np.full(X.shape[0], self.label)
    
    def fit(self, X, y):
        return self
