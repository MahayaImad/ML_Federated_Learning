"""
Serveur FL optimisé pour environnement edge
"""
import tensorflow as tf
import numpy as np


class EdgeFederatedServer:
    def __init__(self, scenario):
        self.scenario = scenario
        self.global_model = None
        self.round_number = 0

    def sync_aggregate(self, device_updates):
        """Agrégation synchrone standard"""
        if not device_updates:
            return 0.0

        # Simulation agrégation simple
        self.round_number += 1
        return min(0.95, 0.1 + (self.round_number * 0.02))

    def async_aggregate(self, device_updates):
        """Agrégation asynchrone pour edge"""
        return self.sync_aggregate(device_updates) * 0.98