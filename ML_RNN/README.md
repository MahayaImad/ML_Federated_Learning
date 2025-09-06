# 🕐 Entraînement RNN sur Datasets Benchmark

Projet pour entraîner des LSTM/GRU sur des datasets benchmark reconnus pour séries temporelles.

## 🚀 Installation

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn yfinance requests
```

## 📝 Usage

### Commandes de base

```bash
# Actions Apple (AAPL) via Yahoo Finance
python train.py --dataset stock --model lstm --epochs 50

# Bitcoin via CoinGecko API  
python train.py --dataset crypto --model gru --epochs 100

# Consommation électrique UCI
python train.py --dataset energy --model lstm --sequence_length 72

# Données météo OpenML
python train.py --dataset weather --model gru --epochs 75

# Prix immobilier Boston (série temporelle)
python train.py --dataset housing --model lstm --epochs 60

# CPU seulement
python train.py --dataset stock --model lstm --epochs 30 --gpu -1
```

### Paramètres disponibles

| Paramètre | Description | Valeurs | Défaut |
|-----------|-------------|---------|--------|
| `--dataset` | Dataset benchmark | `stock`, `crypto`, `energy`, `weather`, `housing` | **Obligatoire** |
| `--model` | Type de RNN | `lstm`, `gru`, `rnn` | **Obligatoire** |
| `--epochs` | Nombre d'époques | Entier | 50 |
| `--batch_size` | Taille des lots | Entier | 32 |
| `--lr` | Taux d'apprentissage | Décimal | 0.001 |
| `--sequence_length` | Longueur séquences | Entier | 50 |
| `--hidden_units` | Unités cachées | Entier | 50 |
| `--dropout` | Taux de dropout | 0.0-1.0 | 0.2 |
| `--gpu` | GPU à utiliser (-1 = CPU) | Entier | 0 |
| `--seed` | Graine aléatoire | Entier | 42 |
| `--verbose` | Mode détaillé | Flag | False |

## 📁 Structure des fichiers générés

```
project/
├── results/                             # results détaillés (.txt)
├── models/                           # Modèles + scalers (.keras, .pkl)
├── visualizations/                   # Courbes d'entraînement (.png)
└── train.py                          # Script principal
```

## 🎯 Datasets benchmark supportés

### 📈 Stock (Actions)
- **Source**: Yahoo Finance API
- **Données**: Apple (AAPL) - 5 dernières années
- **Features**: Prix, volume, moyennes mobiles, RSI, volatilité
- **Fréquence**: Quotidienne

### ₿ Crypto (Cryptomonnaies)  
- **Source**: CoinGecko API (gratuite)
- **Données**: Bitcoin (BTC-USD) - 5 dernières années
- **Features**: Prix, volume, market cap, volatilité, moyennes mobiles
- **Fréquence**: Quotidienne

### ⚡ Energy (Énergie)
- **Source**: UCI Machine Learning Repository
- **Données**: Household Electric Power Consumption (2006-2010)
- **Features**: Puissance globale, réactive, intensité, voltage, sous-compteurs
- **Fréquence**: Horaire (rééchantillonné)

### 🌡️ Weather (Météo)
- **Source**: OpenML/NOAA via scikit-learn
- **Données**: Températures quotidiennes avec patterns réalistes  
- **Features**: Température, humidité, pression, vitesse du vent
- **Fréquence**: Quotidienne

### 🏠 Housing (Immobilier)
- **Source**: Boston Housing (sklearn) + simulation temporelle
- **Données**: Évolution prix immobilier avec cycles réalistes
- **Features**: Prix, taux d'intérêt, chômage, permis construction
- **Fréquence**: Mensuelle

## 🏗️ Architectures optimisées

### LSTM (Long Short-Term Memory)
```
Input(seq_len, features) → LSTM(50) → Dropout → LSTM(25) → Dropout → Dense(25) → Dense(1)
```
**Recommandé pour**: Séries avec dépendances long terme (stock, crypto)

### GRU (Gated Recurrent Unit)  
```
Input(seq_len, features) → GRU(50) → Dropout → GRU(25) → Dropout → Dense(25) → Dense(1)
```
**Recommandé pour**: Plus rapide, performances similaires (energy, weather)

### Simple RNN
```
Input(seq_len, features) → SimpleRNN(50) → Dropout → SimpleRNN(25) → Dropout → Dense(25) → Dense(1)
```
**Recommandé pour**: Baseline de comparaison

## 💡 Conseils par dataset

### Stock/Crypto
- `--sequence_length 60` (3 mois de données)
- `--hidden_units 64` 
- `--dropout 0.2` (marchés volatils)
- `--lr 0.001` (données normalisées)

### Energy/Weather  
- `--sequence_length 72` (3 jours/cycles)
- `--hidden_units 50`
- `--dropout 0.3` (plus de régularisation)
- `--lr 0.001`

### Housing
- `--sequence_length 24` (2 ans de données mensuelles)
- `--hidden_units 32` (moins complexe)
- `--epochs 60` (convergence plus lente)

## 🐛 Dépannage

### Erreurs de téléchargement
```bash
# Stock - problème yfinance
pip install --upgrade yfinance
python train.py --dataset crypto --model lstm  # Alternative

# Crypto - problème réseau
python train.py --dataset stock --model lstm   # Alternative

# Energy - problème UCI/sklearn  
pip install --upgrade scikit-learn requests
```

### Problèmes de performance
```bash
# Overfitting
python train.py --dataset stock --model lstm --dropout 0.4 --epochs 30

# Underfitting  
python train.py --dataset stock --model lstm --hidden_units 100 --epochs 100

# Convergence lente
python train.py --dataset stock --model lstm --lr 0.01 --batch_size 64
```

## 📊 Performances benchmark attendues

| Dataset | Modèle | Époque | MSE | MAE | Temps (GPU) |
|---------|--------|--------|-----|-----|-------------|
| Stock | LSTM | 50 | ~0.01-0.05 | ~0.08-0.15 | ~8 min |
| Crypto | GRU | 100 | ~0.02-0.08 | ~0.10-0.20 | ~12 min |
| Energy | LSTM | 75 | ~0.03-0.10 | ~0.12-0.25 | ~10 min |
| Weather | GRU | 75 | ~0.02-0.06 | ~0.09-0.18 | ~8 min |
| Housing | LSTM | 60 | ~0.05-0.15 | ~0.15-0.30 | ~6 min |

*MSE/MAE sur données normalisées. Performances sur GPU moderne (GTX 1060+)*

## 🔧 Aide

```bash
python train.py --help  # Voir tous les paramètres et exemples
```

## 🚀 Exemples avancés

```bash
# Configuration haute performance (GPU requis)
python train.py --dataset stock --model lstm --epochs 100 --hidden_units 100 --batch_size 64 --sequence_length 90

# Test rapide sur CPU  
python train.py --dataset housing --model gru --epochs 20 --gpu -1 --verbose

# Optimisation pour crypto volatil
python train.py --dataset crypto --model gru --dropout 0.3 --lr 0.0005 --sequence_length 60

# Comparaison modèles sur même dataset
python train.py --dataset energy --model lstm --seed 42 --epochs 50
python train.py --dataset energy --model gru --seed 42 --epochs 50  
python train.py --dataset energy --model rnn --seed 42 --epochs 50
```

## 📚 Sources et références

- **Yahoo Finance**: Actions temps réel via yfinance
- **CoinGecko**: API crypto gratuite et fiable
- **UCI ML Repository**: Datasets académiques validation
- **OpenML**: Plateforme datasets machine learning
- **NOAA**: Données météorologiques officielles US

## ⚠️ Notes importantes

1. **Connexion internet requise** pour téléchargement initial
2. **Données en cache** après première utilisation  
3. **Early stopping** activé (patience=15) pour éviter overfitting
4. **Division temporelle** respectée (pas de data leakage)
5. **Normalisation MinMaxScaler** pour toutes les features