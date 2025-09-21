# 🔄 Comparaison Apprentissage Fédéré vs Autres Méthodes

Projet académique pour comparer l'apprentissage fédéré avec les approches centralisées et distribuées.

## 🚀 Installation

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## 📝 Usage

### Commandes de base

```bash
# FL vs Centralisé sur MNIST
python main.py --dataset mnist --comparison fl_vs_central --epochs 20

# FL vs Distribué sur CIFAR-10  
python main.py --dataset cifar --comparison fl_vs_distributed --epochs 50

# Comparaison complète (non-IID, toutes méthodes)
python main.py --dataset mnist --comparison all --epochs 25 --verbose --plot

# Étude impact distribution des données
python main.py --dataset cifar --comparison all --epochs 30 --iid  # IID
python main.py --dataset cifar --comparison all --epochs 30        # non-IID

# Analyse performance avec plus de clients
python main.py --dataset mnist --comparison all --clients 10 --epochs 20

# CPU seulement
python main.py --dataset mnist --comparison fl_vs_central --gpu -1
```

### Paramètres disponibles

| Paramètre | Description | Valeurs | Défaut |
|-----------|-------------|---------|--------|
| `--dataset` | Dataset à utiliser | `mnist`, `cifar` | **Obligatoire** |
| `--comparison` | Type de comparaison | `fl_vs_central`, `fl_vs_distributed`, `all` | **Obligatoire** |
| `--epochs` | Nombre d'époques | Entier | 20 |
| `--clients` | Nombre de clients FL | Entier | 5 |
| `--batch_size` | Taille des lots | Entier | 32 |
| `--lr` | Taux d'apprentissage | Décimal | 0.001 |
| `--iid` | Distribution IID | Flag | False (non-IID) |
| `--gpu` | GPU à utiliser (-1 = CPU) | Entier | 0 |
| `--verbose` | Mode détaillé | Flag | False |
| `--plot` | Afficher graphiques | Flag | False |

## 📁 Structure des fichiers générés

```
FL-Comparison/
├── results/                             # Résultats détaillés (.txt, .json)
├── models/                           # Modèles entraînés (.keras)
├── visualizations/                   # Graphiques de comparaison (.png)
└── main.py                          # Script principal
```

## 🤖 Méthodes FL implémentées

### FedAvg (Standard)
- **Description**: Agrégation par moyenne pondérée simple
- **Usage**: Baseline de référence
- **Communication**: Modérée

### Cyclic Weighted FL
- **Description**: Sélection cyclique des clients avec pondération dynamique
- **Avantages**: Réduction des biais, adaptation à l'hétérogénéité
- **Communication**: Réduite (sélection partielle)

### Ensemble FL
- **Description**: Multiple modèles FL entraînés sur des sous-ensembles de clients
- **Avantages**: Robustesse, performance améliorée
- **Communication**: Élevée (multiples agrégations)

## 🔬 Types de comparaisons

### FL vs Centralisé
- **Fédéré (FedAvg)**: Entraînement distribué avec agrégation moyenne
- **Centralisé**: Entraînement sur toutes les données réunies
- **Métriques**: Accuracy, temps, communication

### FL vs Distribué
- **Fédéré (FedAvg)**: Agrégation coordonnée par serveur central
- **Distribué**: Entraînement parallèle sans coordination
- **Métriques**: Convergence, stabilité, efficacité

### Comparaison complète (all)
- **Accuracy finale**: Performance sur le test set
- **Temps d'entraînement**: Efficacité computationnelle  
- **Coût de communication**: Échange de paramètres entre clients/serveur
- **Convergence**: Vitesse et stabilité de l'apprentissage
- **Robustesse**: Performance en environnement **non-IID** (défaut)
- **Variance**: Stabilité entre différentes exécutions
- **Ensemble metrics**: Diversité et consensus des modèles (pour FL Ensemble)

## 📊 Métriques évaluées

- **Accuracy finale**: Performance sur le test set
- **Temps d'entraînement**: Efficacité computationnelle  
- **Coût de communication**: Échange de paramètres
- **Convergence**: Vitesse et stabilité
- **Robustesse**: Performance en environnement non-IID


## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| Erreur GPU | `--gpu -1` |
| Lent | `--clients 3 --epochs 10` |
| Données déséquilibrées | `--iid` |

## 📚 Références

1. McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data"
2. Li et al. "Federated Learning: Challenges, Methods, and Future Directions"
3. Kairouz et al. "Advances and Open Problems in Federated Learning"
4. Zhang et al. "CyclicFL: A cyclic model pre-training approach to efficient federated learning."
5. Shlezinger et al. "Collaborative inference via ensembles on the edge."
