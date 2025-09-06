"""
Implémentation de l'algorithme SCAFFOLD
"""
import time
import numpy as np
from .base_aggregator import BaseAggregator, weighted_average, subtract_weights, add_weights
from config import SCAFFOLD_LEARNING_RATE


class ScaffoldAggregator(BaseAggregator):
    """Agrégateur SCAFFOLD avec variables de contrôle"""

    def __init__(self, lr=SCAFFOLD_LEARNING_RATE):
        super().__init__("SCAFFOLD")
        self.lr = lr
        self.server_control = None
        self.client_controls = {}

    def aggregate(self, client_updates, client_weights, global_model):
        """Agrégation SCAFFOLD avec variables de contrôle"""
        start_time = time.time()

        # Extraire les mises à jour et contrôles clients
        model_updates = [update['model_update'] for update in client_updates]
        control_updates = [update['control_update'] for update in client_updates]

        # Moyenne pondérée des mises à jour
        aggregated_update = weighted_average(model_updates, client_weights)
        aggregated_control = weighted_average(control_updates, client_weights)

        # Mise à jour du modèle global
        global_weights = global_model.get_weights()
        new_global_weights = [
            global_w + self.lr * update
            for global_w, update in zip(global_weights, aggregated_update)
        ]

        # Mise à jour du contrôle serveur
        if self.server_control is None:
            self.server_control = [np.zeros_like(w) for w in global_weights]

        self.server_control = add_weights(self.server_control, aggregated_control)

        # Métriques
        comm_cost = self.get_communication_cost(client_updates)
        agg_time = time.time() - start_time

        self.history['communication_costs'].append(comm_cost)
        self.history['aggregation_times'].append(agg_time)

        return new_global_weights

    def prepare_client_update(self, client_id, local_model, global_model):
        """Prépare la mise à jour SCAFFOLD avec contrôles"""
        local_weights = local_model.get_weights()
        global_weights = global_model.get_weights()

        # Initialiser le contrôle client si nécessaire
        if client_id not in self.client_controls:
            self.client_controls[client_id] = [np.zeros_like(w) for w in global_weights]

        # Calculer la mise à jour du modèle
        model_update = subtract_weights(local_weights, global_weights)

        # Calculer la mise à jour du contrôle
        control_update = subtract_weights(
            self.client_controls[client_id],
            self.server_control if self.server_control else [np.zeros_like(w) for w in global_weights]
        )

        return {
            'model_update': model_update,
            'control_update': control_update
        }