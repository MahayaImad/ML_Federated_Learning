"""
Configuration globale pour les expériences d'agrégation fédérée
"""

import os

# Configuration of Hierarchy
CLIENTS = 20
EDGE_SERVERS = 5

# Configuration of training
COMMUNICATION_ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

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