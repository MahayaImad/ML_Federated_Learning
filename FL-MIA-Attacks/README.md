# 🎯 Attaques MIA sur Apprentissage Fédéré

Projet académique pour évaluer la vulnérabilité des modèles fédérés aux attaques d'inférence d'appartenance (MIA).

## 🚀 Installation

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## 📝 Usage

### Commandes de base

```bash
# Attaque Shadow Model (risque moyen)
python main.py --dataset mnist --attack shadow_model --risk_level medium

# Attaque Threshold (risque élevé)
python main.py --dataset cifar --attack threshold --risk_level high --epochs 50

# Toutes les attaques
python main.py --dataset mnist --attack all --risk_level low --verbose --plot

# CPU seulement
python main.py --dataset mnist --attack shadow_model --risk_level medium --gpu -1
```

### Paramètres disponibles

| Paramètre | Description | Valeurs | Défaut |
|-----------|-------------|---------|--------|
| `--dataset` | Dataset à utiliser | `mnist`, `cifar` | **Obligatoire** |
| `--attack` | Type d'attaque MIA | `shadow_model`, `threshold`, `gradient_based`, `all` | **Obligatoire** |
| `--risk_level` | Niveau de risque | `low`, `medium`, `high` | **Obligatoire** |
| `--epochs` | Nombre d'époques | Entier | 30 |
| `--clients` | Nombre de clients FL | Entier | 5 |
| `--target_samples` | Échantillons cibles | Entier | 1000 |
| `--shadow_models` | Modèles shadow | Entier | 5 |
| `--gpu` | GPU à utiliser (-1 = CPU) | Entier | 0 |
| `--verbose` | Mode détaillé | Flag | False |
| `--plot` | Afficher graphiques | Flag | False |

## 📁 Structure des fichiers générés

```
FL-MIA-Attacks/
├── results/                             # Rapports d'attaque (.txt, .json)
├── models/                           # Modèles cibles et shadow (.keras)
├── visualizations/                   # Courbes ROC et métriques (.png)
└── main.py                          # Script principal
```

## 🎯 Types d'attaques MIA

### Shadow Model Attack
- **Principe**: Entraîner des modèles shadow pour imiter le modèle cible
- **Efficacité**: Élevée, mais coûteuse en ressources
- **Détection**: Analyse des probabilités de sortie

### Threshold Attack
- **Principe**: Utiliser un seuil sur les scores de confiance
- **Efficacité**: Modérée, simple à implémenter
- **Détection**: Basée sur la confiance du modèle

### Gradient-Based Attack
- **Principe**: Analyser les gradients pour détecter l'appartenance
- **Efficacité**: Variable selon l'architecture
- **Détection**: Inspection des mises à jour des gradients

## 📊 Niveaux de risque

### 🟢 Low (Faible)
- **Configuration**: Protection renforcée, DP activée
- **Objectif**: Évaluer la robustesse des défenses
- **Seuil d'alerte**: Accuracy > 0.55

### 🟡 Medium (Moyen)
- **Configuration**: Protection standard
- **Objectif**: Cas d'usage réaliste
- **Seuil d'alerte**: Accuracy > 0.60

### 🔴 High (Élevé)
- **Configuration**: Aucune protection
- **Objectif**: Attaque maximale pour tester vulnérabilités
- **Seuil d'alerte**: Accuracy > 0.70

## 📈 Métriques d'évaluation

- **Attack Accuracy**: Capacité à identifier les membres
- **Precision**: Vrais positifs / (Vrais + Faux positifs)
- **Recall**: Vrais positifs / (Vrais positifs + Faux négatifs)  
- **AUC Score**: Aire sous la courbe ROC
- **Évaluation du risque**: Classification automatique

## ⚠️ Évaluation du risque

| Score MIA | Niveau | Interprétation |
|-----------|--------|----------------|
| > 0.70 | 🔴 ÉLEVÉ | Modèle très vulnérable |
| 0.60-0.70 | 🟡 MOYEN | Failles détectées |
| 0.55-0.60 | 🟢 FAIBLE | Protection acceptable |
| < 0.55 | ✅ MINIMAL | Bonne protection |

## 💡 Conseils d'utilisation

- **Recherche défensive?** → `--risk_level low --attack all`
- **Test de vulnérabilité?** → `--risk_level high --attack shadow_model`
- **Évaluation rapide?** → `--attack threshold --target_samples 500`

## 🔒 Considérations éthiques

⚠️ **Usage académique uniquement**
- Évaluation de vulnérabilités pour améliorer la sécurité
- Respect de la confidentialité des données
- Pas d'utilisation malveillante

## 🐛 Dépannage

| Problème | Solution |
|----------|----------|
| Erreur GPU | `--gpu -1` |
| Attaque lente | `--target_samples 500 --shadow_models 3` |
| Manque mémoire | `--batch_size 16` |

## 📚 Références académiques

1. Shokri et al. "Membership Inference Attacks Against Machine Learning Models"
2. Truex et al. "A Hybrid Approach to Privacy-Preserving Federated Learning"
3. Melis et al. "Exploiting Unintended Feature Leakage in Collaborative Learning"
4. Nasr et al. "Comprehensive Privacy Analysis of Deep Learning"
