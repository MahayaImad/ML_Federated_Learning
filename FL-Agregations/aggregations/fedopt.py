"""
Implémentation de l'algorithme FedOpt (FedAdagrad, FedAdam, FedYogi)
"""
import time
import numpy as np
from .base_aggregator import BaseAggregator, weighted_average, subtract_weights
from config import FEDOPT_BETA1, FEDOPT_BETA2, FEDOPT_TAU


class FedOptAggregator(BaseAggregator):
    """Agrégateur FedOpt avec optimiseurs adaptatifs"""

    def __init__(self, optimizer_type="adam", beta1=FEDOPT_BETA1, beta2=FEDOPT_BETA2, tau=FEDOPT_TAU):
        super().__init__(f"FedOpt-{optimizer_type.upper()}")
        self.optimizer_type = optimizer_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = None  # Premier moment
        self.v = None  # Deuxième moment

    def aggregate(self, client_updates, client_weights, global_model):
        """Agrégation avec optimiseur adaptatif"""
        start_time = time.time()

        # Validation des updates
        valid_updates = []
        valid_weights = []

        for i, (update, weight) in enumerate(zip(client_updates, client_weights)):
            try:
                if isinstance(update, list) and all(hasattr(layer, 'shape') for layer in update):
                    valid_updates.append(update)
                    valid_weights.append(weight)
                else:
                    print(f"Update invalide pour client {i}: {type(update)}")
            except Exception as e:
                print(f"Erreur validation client {i}: {e}")
                continue

        if not valid_updates:
            print("Aucune mise à jour valide pour FedOpt")
            return global_model.get_weights()

        # Moyenne pondérée des mises à jour valides
        aggregated_update = weighted_average(valid_updates, valid_weights)

        # Initialiser les moments si nécessaire
        if self.m is None:
            self.m = [np.zeros_like(update) for update in aggregated_update]
            self.v = [np.zeros_like(update) for update in aggregated_update]

        # Mise à jour des moments et application
        global_weights = global_model.get_weights()
        new_global_weights = []

        for i, (global_w, update) in enumerate(zip(global_weights, aggregated_update)):
            # Mise à jour des moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * update

            if self.optimizer_type == "adam":
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (update ** 2)
                # Correction de bias
                m_hat = self.m[i] / (1 - self.beta1 ** (self.round_number + 1))
                v_hat = self.v[i] / (1 - self.beta2 ** (self.round_number + 1))
                # Mise à jour
                new_w = global_w + self.tau * m_hat / (np.sqrt(v_hat) + 1e-8)

            elif self.optimizer_type == "adagrad":
                self.v[i] += update ** 2
                new_w = global_w + self.tau * update / (np.sqrt(self.v[i]) + 1e-8)

            elif self.optimizer_type == "yogi":
                self.v[i] = self.v[i] - (1 - self.beta2) * np.sign(self.v[i] - update ** 2) * (update ** 2)
                new_w = global_w + self.tau * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)

            new_global_weights.append(new_w)

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
        """Même logique que FedAvg"""
        local_weights = local_model.get_weights()
        global_weights = global_model.get_weights()
        client_update = subtract_weights(local_weights, global_weights)
        return client_update

    def reset(self):
        """Reset avec remise à zéro des moments"""
        super().reset()
        self.m = None
        self.v = None


class FedAdamAggregator(FedOptAggregator):
    """FedAdam spécialisé"""

    def __init__(self, **kwargs):
        super().__init__(optimizer_type="adam", **kwargs)


class FedAdagradAggregator(FedOptAggregator):
    """FedAdagrad spécialisé"""

    def __init__(self, **kwargs):
        super().__init__(optimizer_type="adagrad", **kwargs)


class FedYogiAggregator(FedOptAggregator):
    """FedYogi spécialisé"""

    def __init__(self, **kwargs):
        super().__init__(optimizer_type="yogi", **kwargs)