# Système d'Agrégation Fédérée (FL-Agregations)

## 📋 Description

Ce projet implémente différentes méthodes d'agrégation pour l'apprentissage fédéré sur le dataset CIFAR-10. Il compare les performances de plusieurs algorithmes d'agrégation incluant FedAvg, FedProx, SCAFFOLD, et des méthodes sécurisées avec confidentialité différentielle.

## 🎯 Objectifs Académiques

- Étudier les différentes stratégies d'agrégation en apprentissage fédéré
- Comparer les performances en environnements IID et non-IID
- Analyser l'impact de la sécurité et de la confidentialité différentielle
- Évaluer les coûts de communication et temps de convergence

## 🏗️ Architecture

```
FL-Agregations/
├── aggregations/           # Implémentations des algorithmes d'agrégation
│   ├── __init__.py
│   ├── base_aggregator.py  # Classe abstraite de base
│   ├── fedavg.py          # FedAvg standard
│   ├── fedprox.py         # FedProx avec régularisation proximale
│   ├── scaffold.py        # SCAFFOLD avec variables de contrôle
│   ├── fedopt.py          # FedOpt (Adam, Adagrad, Yogi)
│   └── secure_aggregation.py # Agrégation sécurisée + DP
├── experiments/           # Scripts d'expérimentation
│   ├── __init__.py
│   └── run_comparison.py  # Comparaison des méthodes
├── client.py             # Simulation des clients fédérés
├── server.py             # Serveur fédéré
├── models.py             # Définitions des modèles CNN
├── data_preparation.py   # Préparation des données CIFAR-10
├── config.py             # Configuration globale
└── main.py               # Point d'entrée principal
```

## 🚀 Installation

### Prérequis
```bash
Python >= 3.8
TensorFlow >= 2.10
NumPy >= 1.21
Scikit-learn >= 1.0
```

### Installation des dépendances
```bash
pip install tensorflow numpy scikit-learn matplotlib
```

## 📊 Utilisation

### Entraînement avec une méthode spécifique
```bash
# FedAvg avec distribution IID
python main.py --method fedavg --iid

# FedProx avec distribution non-IID
python main.py --method fedprox

# SCAFFOLD avec modèle robuste
python main.py --method scaffold --model robust
```

### Comparaison de toutes les méthodes
```bash
# Comparaison complète
python main.py --method compare --iid

# Comparaison en environnement non-IID
python main.py --method compare
```

### Options disponibles
- `--method`: `fedavg`, `fedprox`, `scaffold`, `compare`
- `--iid`: Distribution IID des données (défaut: non-IID)
- `--model`: `standard`, `lightweight`, `robust`

## 🔬 Méthodes d'Agrégation Implémentées

### 1. **FedAvg** (Baseline)
- Moyenne pondérée simple des mises à jour clients
- Algorithme de référence en apprentissage fédéré

### 2. **FedProx**
- Terme de régularisation proximale pendant l'entraînement local
- Améliore la robustesse en environnements hétérogènes

### 3. **SCAFFOLD**
- Variables de contrôle pour corriger la dérive des clients
- Accélère la convergence en réduisant la variance

### 4. **FedOpt** (Adam, Adagrad, Yogi)
- Optimiseurs adaptatifs côté serveur
- Améliore la convergence avec moments adaptatifs

### 5. **Agrégation Sécurisée**
- Confidentialité différentielle (ε,δ)-DP
- Simulation d'agrégation sécurisée multi-parties

## ⚙️ Configuration

Le fichier `config.py` contient tous les hyperparamètres :

```python
# Données et clients
CLIENTS = 5                    # Nombre de clients
SIZE_PER_CLIENT = 3000        # Échantillons par client

# Entraînement
COMMUNICATION_ROUNDS = 50      # Tours de communication
LOCAL_EPOCHS = 3              # Époques locales par client
LEARNING_RATE = 0.001         # Taux d'apprentissage

# Algorithmes spécifiques
FEDPROX_MU = 0.01             # Régularisation FedProx
DIFFERENTIAL_PRIVACY_EPSILON = 1.0  # Budget ε pour DP
```

## 📈 Métriques Évaluées

- **Précision finale** : Performance du modèle global
- **Vitesse de convergence** : Rounds nécessaires pour converger
- **Coût de communication** : Nombre total de paramètres échangés
- **Temps d'agrégation** : Efficacité computationnelle
- **Stabilité** : Variance des performances

## 🔒 Aspects Sécuritaires

### Confidentialité Différentielle
- Clipping des gradients avec norme L2
- Ajout de bruit gaussien calibré (ε,δ)-DP
- Protection contre l'inférence d'informations privées

### Robustesse
- Détection d'attaques par analyse des normes
- Seuil minimum de participation
- Validation des mises à jour clients

## 📋 Résultats Typiques

```
=== RÉSULTATS DE COMPARAISON ===
FedAvg: Précision = 0.7245
FedProx: Précision = 0.7389
SCAFFOLD: Précision = 0.7456
```

## 🔧 Extensions Possibles

1. **Nouveaux algorithmes** : FedNova, FedBN, LAG
2. **Attaques et défenses** : Attaques byzantines, agrégation robuste
3. **Optimisations** : Compression, quantification
4. **Hétérogénéité** : Modèles différents par client

## 🐛 Dépannage

### Erreurs courantes
```bash
# Erreur de mémoire
ResourceExhaustedError
# Solution: Réduire BATCH_SIZE ou SIZE_PER_CLIENT
```

## 📚 Références

1. McMahan, H. B. et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
2. Li, T. et al. "Federated Optimization in Heterogeneous Networks" (FedProx)  
3. Karimireddy, S. P. et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
4. Reddi, S. et al. "Adaptive Federated Optimization" (FedOpt)


---

*Développé par MAHAYA IMAD dans le cadre d'un projet académique sur l'apprentissage fédéré*