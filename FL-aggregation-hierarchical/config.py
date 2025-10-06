"""
Configuration globale pour les expériences d'agrégation fédérée
"""
import os

# Configuration des données
CLIENTS = 5
SIZE_PER_CLIENT = 3000
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
NUM_CLASSES = 10

# Configuration d'entraînement
COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Configuration des agrégations
FEDPROX_MU = 0.01
SCAFFOLD_LEARNING_RATE = 1.0
FEDOPT_BETA1 = 0.9
FEDOPT_BETA2 = 0.99
FEDOPT_TAU = 1e-3

# Paramètres MOON
MOON_TEMPERATURE = 0.5  # Température pour contrastive loss
MOON_MU = 1.0          # Coefficient de pondération

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "visualizations")

# Créer les dossiers
for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Labels CIFAR-10
CIFAR_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Datasets supportés
SUPPORTED_DATASETS = ['cifar10', 'mnist']

# Configuration par dataset
DATASET_CONFIG = {
    'cifar10': {
        'input_shape': (32, 32, 3),
        'num_classes': 10,
        'model_function': 'create_cifar10_cnn'
    },
    'mnist': {
        'input_shape': (28, 28, 1),
        'num_classes': 10,
        'model_function': 'create_mnist_cnn'
    }
}