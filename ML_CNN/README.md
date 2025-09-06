# 🧠 Entraînement de Modèles CNN

Projet simple pour entraîner des CNN sur MNIST et CIFAR-10 avec interface en ligne de commande.

## 🚀 Installation

```bash
pip install tensorflow numpy matplotlib
```

## 📝 Usage

### Commandes de base

```bash
# MNIST avec CNN (recommandé pour débuter)
python train.py --dataset mnist --model cnn --epochs 10

# CIFAR-10 avec CNN  
python train.py --dataset cifar --model cnn --epochs 50 --lr 0.001

# MNIST avec MLP simple
python train.py --dataset mnist --model mlp --epochs 20

# CPU seulement
python train.py --dataset mnist --model cnn --epochs 10 --gpu -1
```

### Paramètres disponibles

| Paramètre | Description | Valeurs | Défaut |
|-----------|-------------|---------|--------|
| `--dataset` | Dataset à utiliser | `mnist`, `cifar` | **Obligatoire** |
| `--model` | Type de modèle | `cnn`, `mlp` | **Obligatoire** |
| `--epochs` | Nombre d'époques | Entier | 10 |
| `--batch_size` | Taille des lots | Entier | 64 |
| `--lr` | Taux d'apprentissage | Décimal | 0.01 |
| `--gpu` | GPU à utiliser (-1 = CPU) | Entier | 0 |
| `--seed` | Graine aléatoire | Entier | 42 |
| `--verbose` | Mode détaillé | Flag | False |

## 📁 Structure des fichiers générés

```
project/
├── results/                          # results détaillés (.txt)
├── models/                           # Modèles sauvegardés (.keras)
├── visualizations/                   # Courbes d'entraînement (.png)
└── train.py                          # Script principal
```

## 🎯 Datasets supportés

- **MNIST**: Images N&B 28x28, chiffres 0-9, 60k train + 10k test
- **CIFAR-10**: Images couleur 32x32, 10 classes d'objets, 50k train + 10k test

## 💡 Conseils

- **Débutant?** → `mnist` + `cnn` + `10 époques`
- **Perte oscille?** → Diminuer `--lr` (ex: 0.001)  
- **Lent?** → Augmenter `--batch_size` ou utiliser GPU
- **Pas de GPU?** → `--gpu -1`

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| Erreur GPU | `--gpu -1` |
| Manque mémoire | `--batch_size 32` |
| Perte ne baisse pas | `--lr 0.001` |

## 📊 Performances attendues

| Dataset | Modèle | Époques | Précision | Temps (GPU) |
|---------|--------|---------|-----------|-------------|
| MNIST | CNN | 10 | ~98% | ~2 min |
| CIFAR-10 | CNN | 50 | ~75% | ~10 min |

## 🔧 Aide

```bash
python train.py --help  # Voir tous les paramètres
```