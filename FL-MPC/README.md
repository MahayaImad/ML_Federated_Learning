# FL-MPC: Federated Learning with Secure Multi-Party Computation

## Description

Ce projet implémente l'apprentissage fédéré avec calcul multipartite sécurisé (MPC) et le compare avec l'apprentissage fédéré classique. Il utilise le dataset CIFAR-10 pour l'évaluation des performances.

Cette Methode est bien expliquée dans cette vidéo : https://www.youtube.com/watch?v=zrmMmq9N9FY

## Objectifs

- **Sécurité renforcée** : Protection des mises à jour des clients via secret sharing
- **Résistance aux attaques** : Tolérance aux clients byzantins
- **Comparaison empirique** : Analyse des performances FL classique vs FL-MPC
- **Recherche académique** : Implémentation simple mais correcte des concepts MPC

## Structure du projet

```
FL-MPC/
├── README.md
├── config.py                 # Configuration globale
├── data_preparation.py       # Préparation des données (copié de FL_Agregations)
├── models.py                 # Modèles CNN (copié de FL_Agregations)
├── client.py                 # Client MPC
├── server.py                 # Serveur MPC
├── main.py                   # Point d'entrée principal
├── mpc/                      # Modules MPC
│   ├── __init__.py
│   ├── secret_sharing.py     # Partage de secret de Shamir
│   ├── secure_aggregation.py # Agrégation sécurisée
│   └── mpc_protocols.py      # Protocoles MPC additionnels
├── comparison/               # Outils de comparaison
│   ├── __init__.py
│   └── fl_vs_mpc.py          # Comparaison FL vs FL-MPC
├── plots/                    # Graphiques générés
└── results/                  # résultats d'exécution
```

## Prérequis

### Dépendances système
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn

### Installation
```bash
# Installer les dépendances
pip install tensorflow numpy matplotlib scikit-learn

# Vérifier que FL_Agregations est disponible dans le répertoire parent
ls ../FL_Agregations/
```

## Concepts techniques

### Secret Sharing de Shamir
- **Seuil** : Minimum 3 clients pour reconstruction
- **Corps fini** : Calculs modulo 2³¹-1
- **Précision** : 16 bits pour nombres à virgule fixe

### Agrégation sécurisée
1. **Partage** : Chaque client partage ses mises à jour
2. **Calcul** : Moyenne pondérée sur parts secrètes
3. **Reconstruction** : Révélation du résultat agrégé uniquement

### Modèles de menace
- **Semi-honnête** : Clients suivent le protocole mais curieux
- **Byzantin** : Clients malveillants avec comportement arbitraire
- **Tolérance** : Jusqu'à 1 client byzantin par défaut

## Utilisation

### Comparaison rapide
```bash
python main.py --iid --rounds 20 --plot
```

### Options disponibles
```bash
python main.py --help

Options:
  --iid          Distribution IID des données (défaut: non-IID)
  --rounds N     Nombre de tours d'entraînement (défaut: 30)
  --plot         Afficher les graphiques de comparaison
  --byzantine N  Nombre de clients byzantins à simuler
```

### Exemples d'utilisation

**Comparaison standard (non-IID)**
```bash
python main.py --rounds 50 --plot
```

**Test avec données IID**
```bash
python main.py --iid --rounds 30 --plot
```

**Simulation d'attaques byzantines**
```bash
python main.py --byzantine 1 --rounds 40 --plot
```

## Métriques évaluées

### Performance
- **Accuracy** : Accuracy sur le test set
- **Convergence** : Vitesse de convergence
- **Stabilité** : Variance des performances

### Efficacité
- **Temps d'exécution** : Par tour et total
- **Communication** : Overhead MPC vs FL classique
- **Scalabilité** : Performance selon nombre de clients

### Sécurité
- **Résistance** : Face aux attaques byzantines
- **Confidentialité** : Protection des données clients
- **Robustesse** : Stabilité avec clients défaillants

## Configuration

### Paramètres principaux (`config.py`)
```python
# Données
CLIENTS = 5
SIZE_PER_CLIENT = 3000

# Entraînement
COMMUNICATION_ROUNDS = 50
LOCAL_EPOCHS = 3

# MPC
MPC_THRESHOLD = 3
MPC_FIELD_SIZE = 2**31 - 1
BYZANTINE_TOLERANCE = 1
```

### Personnalisation
Modifier `config.py` pour ajuster :
- Nombre de clients et taille des données
- Paramètres de sécurité MPC
- Seuils de tolérance byzantine

## Résultats attendus

### FL-MPC vs FL Classique

**Avantages FL-MPC :**
- ✅ Sécurité renforcée
- ✅ Résistance aux attaques
- ✅ Confidentialité garantie

**Inconvénients FL-MPC :**
- ❌ Overhead de communication (3-5x)
- ❌ Temps d'exécution supérieur (2-4x)
- ❌ Complexité d'implémentation

### Métriques typiques
```
=== RÉSULTATS FINAUX ===
FL Classique - Accuracy finale: 0.7245
FL-MPC - Accuracy finale: 0.7156
FL Classique - Temps moyen/tour: 12.34s
FL-MPC - Temps moyen/tour: 45.67s
Overhead MPC: 3.2x communication
```

## Architecture technique

### Client MPC (`client.py`)
```python
class MPCClient:
    - get_secure_update()    # Prépare update pour MPC
    - simulate_byzantine()   # Simulation d'attaques
    - validate_shares()      # Validation des parts
```

### Serveur MPC (`server.py`)
```python
class MPCServer:
    - secure_aggregate()     # Agrégation MPC
    - byzantine_detection()  # Détection d'anomalies
    - threshold_check()      # Vérification seuil
```

### Modules MPC (`mpc/`)
```python
secret_sharing.py:
    - ShamirSecretSharing   # Partage de secret
    - polynomial_eval()     # Évaluation polynômiale
    - lagrange_interpolate() # Reconstruction

secure_aggregation.py:
    - MPCAggregator         # Agrégateur principal
    - share_weights()       # Partage des poids
    - compute_average()     # Moyenne sécurisée
```

## Références académiques

1. **Shamir, A.** (1979). How to share a secret. CACM.
2. **Bonawitz, K. et al.** (2017). Practical secure aggregation for privacy-preserving machine learning. CCS.
3. **McMahan, B. et al.** (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS.
4. **Xu, R. et al.** (2019). Towards practical differentially private convex optimization. S&P.

---