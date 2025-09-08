"""
Implémentation de l'algorithme FedAvg
"""
import time
import numpy as np
from .base_aggregator import BaseAggregator, weighted_average, subtract_weights


class FedAvgAggregator(BaseAggregator):
    """Agrégateur FedAvg standard"""

    def __init__(self):
        super().__init__("FedAvg")

    def aggregate(self, client_updates, client_weights, global_model):
        """
        Agrège les mises à jour avec FedAvg

        Args:
            client_updates: Liste des deltas de poids des clients
            client_weights: Tailles des datasets des clients
            global_model: Modèle global actuel

        Returns:
            new_global_weights: Nouveaux poids du modèle global
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

        # Calculer le delta
        client_update = subtract_weights(local_weights, global_weights)

        return client_update


class FedAvgMomentumAggregator(BaseAggregator):
    """FedAvg avec momentum sur le serveur"""

    def __init__(self, momentum=0.9):
        super().__init__("FedAvg-Momentum")
        self.momentum = momentum
        self.velocity = None

    def aggregate(self, client_updates, client_weights, global_model):
        """Agrégation avec momentum"""
        start_time = time.time()

        # Moyenne pondérée des mises à jour
        aggregated_update = weighted_average(client_updates, client_weights)

        # Initialiser la vélocité si nécessaire
        if self.velocity is None:
            self.velocity = [np.zeros_like(update) for update in aggregated_update]

        # Mise à jour avec momentum
        for i in range(len(self.velocity)):
            self.velocity[i] = (self.momentum * self.velocity[i] +
                                (1 - self.momentum) * aggregated_update[i])

        # Application au modèle global
        global_weights = global_model.get_weights()
        new_global_weights = [
            global_w + velocity for global_w, velocity in zip(global_weights, self.velocity)
        ]

        # Métriques
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
        """Même logique que FedAvg standard"""
        return FedAvgAggregator.prepare_client_update(self, client_id, local_model, global_model)

    def reset(self):
        """Reset avec remise à zéro de la vélocité"""
        super().reset()
        self.velocity = None