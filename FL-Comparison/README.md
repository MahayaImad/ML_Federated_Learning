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

# Comparaison complète
python main.py --dataset mnist --comparison all --epochs 30 --verbose --plot

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

## 🔬 Types de comparaisons

### FL vs Centralisé
- **Fédéré**: Entraînement distribué sur plusieurs clients
- **Centralisé**: Entraînement sur toutes les données réunies
- **Métriques**: Accuracy, temps, communication

### FL vs Distribué
- **Fédéré**: Agrégation coordonnée par serveur central
- **Distribué**: Entraînement parallèle sans coordination
- **Métriques**: Convergence, stabilité, efficacité

### Comparaison complète
- **Tous les types**: FL, Centralisé, Distribué, variants
- **Analyse complète**: Performance, ressources, robustesse

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
