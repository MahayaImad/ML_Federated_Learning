"""
Configuration pour FL-MPC
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
COMMUNICATION_ROUNDS = 50
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Configuration des agrégations
FEDPROX_MU = 0.01
SCAFFOLD_LEARNING_RATE = 1.0
FEDOPT_BETA1 = 0.9
FEDOPT_BETA2 = 0.99
FEDOPT_TAU = 1e-3

# Configuration de sécurité
SECURE_AGGREGATION_THRESHOLD = 3
DIFFERENTIAL_PRIVACY_EPSILON = 1.0
DIFFERENTIAL_PRIVACY_DELTA = 1e-5

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Créer les dossiers
for directory in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Labels CIFAR-10
CIFAR_LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Paramètres spécifiques MPC
MPC_THRESHOLD = 3  # Seuil pour secret sharing
MPC_FIELD_SIZE = 2**31 - 1  # Taille du corps fini
MPC_PRECISION = 16  # Précision pour nombres à virgule fixe

# Sécurité
BYZANTINE_TOLERANCE = 1  # Nombre de clients byzantins tolérés