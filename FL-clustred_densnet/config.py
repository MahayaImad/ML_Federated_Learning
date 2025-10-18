"""
Global configuration for federated aggregation experiments
Includes support for DenseNet transfer learning on CIFAR-10
"""

import os

# Configuration of Hierarchy
CLIENTS = 20
EDGE_SERVERS = 5

# Configuration of training
COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  # Will be overridden for transfer learning (1e-4 for CIFAR-10)

# Transfer Learning Configuration (CIFAR-10)
USE_TRANSFER_LEARNING_CIFAR10 = True  # Automatically use DenseNet for CIFAR-10
DENSENET_FROZEN_LAYERS = True  # Freeze base layers
TRANSFER_LEARNING_LR = 1e-4  # Lower learning rate for fine-tuning

# Saving clients results
SAVE_CLIENTS_STATS = True
VERBOSE = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "visualizations")

# Create Folders
for directory in [RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)