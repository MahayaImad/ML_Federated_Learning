"""
Configuration pour comparaisons FL
"""
import os

# Paramètres par défaut comparaison
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_CLIENTS = 5
DEFAULT_SEED = 42

# Types de comparaisons
COMPARISON_TYPES = ['fl_vs_central', 'fl_vs_distributed', 'all']

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

# Labels des classes
MNIST_LABELS = list(range(10))
CIFAR_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Métriques de comparaison
COMPARISON_METRICS = [
    'final_accuracy',
    'training_time',
    'communication_cost',
    'convergence_round'
]

# Configuration des méthodes d'apprentissage
LEARNING_METHODS = {
    'centralized': {
        'description': 'Entraînement sur toutes les données réunies',
        'communication_cost': 0,
        'privacy_level': 'low'
    },
    'federated': {
        'description': 'Apprentissage fédéré avec agrégation FedAvg',
        'communication_cost': 'variable',
        'privacy_level': 'medium'
    },
    'distributed': {
        'description': 'Entraînement parallèle sans coordination',
        'communication_cost': 0,
        'privacy_level': 'high'
    }
}

# Messages d'aide
HELP_MESSAGES = {
    'dataset': 'Dataset à utiliser (mnist: images 28x28 N&B, cifar: images 32x32 couleur)',
    'comparison': 'Type de comparaison (fl_vs_central: FL vs centralisé, fl_vs_distributed: FL vs distribué, all: toutes)',
    'clients': 'Nombre de clients pour l\'apprentissage fédéré',
    'iid': 'Distribution IID des données (par défaut: non-IID plus réaliste)',
    'epochs': 'Nombre d\'époques d\'entraînement'
}

# Configurations recommandées par dataset
DATASET_CONFIGS = {
    'mnist': {
        'recommended_epochs': 15,
        'recommended_clients': 5,
        'expected_accuracy': {
            'centralized': 0.99,
            'federated': 0.97,
            'distributed': 0.95
        }
    },
    'cifar': {
        'recommended_epochs': 30,
        'recommended_clients': 5,
        'expected_accuracy': {
            'centralized': 0.75,
            'federated': 0.70,
            'distributed': 0.65
        }
    }
}