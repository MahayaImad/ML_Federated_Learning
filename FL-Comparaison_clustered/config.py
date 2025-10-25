"""
Configuration parameters for FL Comparison Framework
"""

# Default hyperparameters
CLIENTS = 20
COMMUNICATION_ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32

# Data distribution
ALPHA = 0.5  # Dirichlet alpha for non-IID

# Agglomerative parameters
JS_THRESHOLD = 0.5
SELECTION_RATIO = 0.3

# FedProx parameters
MU = 0.01  # Proximal term

# IFCA parameters
NUM_CLUSTERS = 5

# Output settings
SAVE_CLIENTS_STATS = True
VERBOSE = False

# Communication cost parameters (in KB)
MODEL_SIZE_MNIST = 500  # Approximate model size for MNIST in KB
MODEL_SIZE_CIFAR10 = 1000  # Approximate model size for CIFAR10 in KB
MODEL_SIZE_MALNET = 800  # Approximate model size for MALNET in KB

# Dataset configurations
DATASET_CONFIG = {
    'mnist': {
        'input_shape': (28, 28, 1),
        'num_classes': 10,
        'model_type': 'cnn',
        'model_size_kb': MODEL_SIZE_MNIST
    },
    'cifar10': {
        'input_shape': (32, 32, 3),
        'num_classes': 10,
        'model_type': 'cnn',
        'model_size_kb': MODEL_SIZE_CIFAR10
    },
    'malnet': {
        'input_shape': None,  # Will be determined from data
        'num_classes': 20,  # Malware families
        'model_type': 'dnn',
        'model_size_kb': MODEL_SIZE_MALNET
    }
}


def get_dataset_config(dataset_name):
    """Get configuration for a specific dataset"""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIG[dataset_name]


def get_model_size(dataset_name):
    """Get model size in KB for a specific dataset"""
    config = get_dataset_config(dataset_name)
    return config['model_size_kb']