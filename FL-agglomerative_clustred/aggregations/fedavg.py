import time
import numpy as np
from .base_aggregator import BaseAggregator, weighted_average, subtract_weights


class FedAvgAggregator(BaseAggregator):
    """Standard FedAvg aggregator"""

    def __init__(self):
        super().__init__("FedAvg")

    def aggregate(self, client_updates, client_weights, global_model):
        """
        Aggregate client updates using FedAvg

        Args:
            client_updates: List of client weight deltas
            client_weights: Data sizes for weighted averaging
            global_model: Current global model

        Returns:
            new_global_weights: Updated global model weights
        """
        start_time = time.time()

        # Calcul de la moyenne pondérée des mises à jour
        aggregated_update = weighted_average(client_updates, client_weights)

        # Application de la mise à jour au modèle global
        global_weights = global_model.get_weights()
        new_global_weights = [
            global_w + update for global_w, update in zip(global_weights, aggregated_update)
        ]

        # Enregistrement des métriques
        comm_cost = self.get_communication_cost(client_updates)
        agg_time = time.time() - start_time

        max_history = 100
        if len(self.history['communication_costs']) >= max_history:
            self.history['communication_costs'].pop(0)
        if len(self.history['aggregation_times']) >= max_history:
            self.history['aggregation_times'].pop(0)

        self.history['communication_costs'].append(comm_cost)
        self.history['aggregation_times'].append(agg_time)

        return new_global_weights

    def prepare_client_update(self, client_id, local_model, global_model):
        """
        Prépare la mise à jour du client (delta des poids)

        Args:
            client_id: ID du client
            local_model: Modèle local entraîné
            global_model: Modèle global

        Returns:
            client_update: Delta des poids (local - global)
        """
        local_weights = local_model.get_weights()
        global_weights = global_model.get_weights()

        # Calculate weight delta
        client_update = subtract_weights(local_weights, global_weights)

        return client_update
