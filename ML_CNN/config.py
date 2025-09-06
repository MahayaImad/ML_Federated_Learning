"""
Configuration par défaut
"""

import os

# Paramètres par défaut
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MOMENTUM = 0.9
DEFAULT_SEED = 42

# Dossiers (structure FL-MIA)
MODEL_DIR = "models"
LOG_DIR = "results"
VISUALIZATION_DIR = "visualizations"

# Créer les dossiers automatiquement
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Datasets supportés
SUPPORTED_DATASETS = ['mnist', 'cifar']
SUPPORTED_MODELS = ['cnn', 'mlp']

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