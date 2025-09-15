"""
Configuration pour attaques MIA
"""
import os

# Paramètres par défaut MIA
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_CLIENTS = 5
DEFAULT_TARGET_SAMPLES = 1000
DEFAULT_SHADOW_MODELS = 5
DEFAULT_SEED = 42

# Niveaux de risque
RISK_LEVELS = ['low', 'medium', 'high']

# Types d'attaques
ATTACK_TYPES = ['shadow_model', 'threshold', 'gradient_based', 'all']

# Dossiers (structure cohérente)
MODELS_DIR = "models"
RESULTS_DIR = "results"
VISUALIZATIONS_DIR = "visualizations"

# Créer les dossiers automatiquement
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Datasets supportés
SUPPORTED_DATASETS = ['mnist', 'cifar']

# Classes par dataset
DATASET_CLASSES = {
    'mnist': 10,
    'cifar': 10
}

# Seuils de risque pour les attaques
RISK_THRESHOLDS = {
    'low': {
        'accuracy_threshold': 0.55,
        'warning_message': "Protection acceptable"
    },
    'medium': {
        'accuracy_threshold': 0.60,
        'warning_message': "Failles de confidentialité détectées"
    },
    'high': {
        'accuracy_threshold': 0.70,
        'warning_message': "Modèle vulnérable aux attaques MIA"
    }
}

# Configuration défenses par niveau de risque
DEFENSE_CONFIG = {
    'low': {
        'dropout_rate': 0.6,
        'noise_scale': 0.1,
        'dp_enabled': True,
        'epsilon': 1.0
    },
    'medium': {
        'dropout_rate': 0.4,
        'noise_scale': 0.05,
        'dp_enabled': False,
        'epsilon': 5.0
    },
    'high': {
        'dropout_rate': 0.2,
        'noise_scale': 0.0,
        'dp_enabled': False,
        'epsilon': float('inf')
    }
}

# Labels des classes
MNIST_LABELS = list(range(10))
CIFAR_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Métriques d'évaluation MIA
MIA_METRICS = [
    'attack_accuracy',
    'precision',
    'recall',
    'auc_score',
    'attack_time'
]

# Messages d'aide
HELP_MESSAGES = {
    'dataset': 'Dataset à utiliser (mnist: 28x28 N&B, cifar: 32x32 couleur)',
    'attack': 'Type d\'attaque MIA à effectuer',
    'risk_level': 'Niveau de risque (low: défenses activées, medium: partielles, high: aucune)',
    'target_samples': 'Nombre d\'échantillons pour tester l\'appartenance',
    'shadow_models': 'Nombre de modèles shadow pour l\'attaque (plus = plus précis mais plus lent)'
}