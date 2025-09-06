"""
Implémentation de l'agrégation sécurisée
"""
import time
import numpy as np
from .base_aggregator import BaseAggregator, weighted_average
from config import SECURE_AGGREGATION_THRESHOLD, DIFFERENTIAL_PRIVACY_EPSILON, DIFFERENTIAL_PRIVACY_DELTA


class SecureAggregator(BaseAggregator):
    """Agrégateur avec sécurité et confidentialité différentielle"""

    def __init__(self, use_dp=True, use_secure_sum=True,
                 epsilon=DIFFERENTIAL_PRIVACY_EPSILON,
                 delta=DIFFERENTIAL_PRIVACY_DELTA,
                 threshold=SECURE_AGGREGATION_THRESHOLD):
        super().__init__("SecureAggregation")
        self.use_dp = use_dp
        self.use_secure_sum = use_secure_sum
        self.epsilon = epsilon
        self.delta = delta
        self.threshold = threshold

    def aggregate(self, client_updates, client_weights, global_model):
        """Agrégation sécurisée avec confidentialité différentielle"""
        start_time = time.time()

        # Vérifier le seuil de participation
        if len(client_updates) < self.threshold:
            print(f"Pas assez de clients ({len(client_updates)} < {self.threshold})")
            return global_model.get_weights()

        # Simulation d'agrégation sécurisée
        if self.use_secure_sum:
            aggregated_update = self._secure_sum_aggregation(client_updates, client_weights)
        else:
            aggregated_update = weighted_average(client_updates, client_weights)

        # Application de la confidentialité différentielle
        if self.use_dp:
            aggregated_update = self._add_differential_privacy_noise(aggregated_update)

        # Mise à jour du modèle global
        global_weights = global_model.get_weights()
        new_global_weights = [
            global_w + update for global_w, update in zip(global_weights, aggregated_update)
        ]

        # Métriques
        comm_cost = self.get_communication_cost(client_updates)
        agg_time = time.time() - start_time

        self.history['communication_costs'].append(comm_cost)
        self.history['aggregation_times'].append(agg_time)

        return new_global_weights

    def _secure_sum_aggregation(self, client_updates, client_weights):
        """Simulation d'une somme sécurisée"""
        # Dans une vraie implémentation, ceci utiliserait des techniques cryptographiques
        # Ici, on simule avec du bruit minimal
        aggregated = weighted_average(client_updates, client_weights)

        # Ajouter un petit bruit pour simuler l'overhead cryptographique
        for i, update in enumerate(aggregated):
            noise = np.random.normal(0, 1e-6, update.shape)
            aggregated[i] = update + noise

        return aggregated

    def _add_differential_privacy_noise(self, aggregated_update):
        """Ajoute du bruit pour la confidentialité différentielle"""
        # Calcul de la sensibilité (simplifiée)
        sensitivity = self._calculate_sensitivity(aggregated_update)

        # Bruit Gaussien pour (epsilon, delta)-DP
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon

        noisy_update = []
        for update in aggregated_update:
            noise = np.random.normal(0, sigma, update.shape)
            noisy_update.append(update + noise)

        return noisy_update

    def _calculate_sensitivity(self, aggregated_update):
        """Calcule la sensibilité L2 globale"""
        total_norm = 0
        for update in aggregated_update:
            total_norm += np.linalg.norm(update) ** 2
        return np.sqrt(total_norm)

    def prepare_client_update(self, client_id, local_model, global_model):
        """Prépare la mise à jour avec clipping potentiel"""
        local_weights = local_model.get_weights()
        global_weights = global_model.get_weights()

        # Calculer la mise à jour
        client_update = [local_w - global_w for local_w, global_w in zip(local_weights, global_weights)]

        # Clipping pour la confidentialité différentielle
        if self.use_dp:
            client_update = self._clip_update(client_update)

        return client_update

    def _clip_update(self, client_update, clip_norm=1.0):
        """Clipe la norme de la mise à jour"""
        # Calculer la norme L2
        total_norm = 0
        for update in client_update:
            total_norm += np.linalg.norm(update) ** 2
        total_norm = np.sqrt(total_norm)

        # Clipper si nécessaire
        if total_norm > clip_norm:
            clipped_update = []
            for update in client_update:
                clipped_update.append(update * clip_norm / total_norm)
            return clipped_update

        return client_update


class DifferentialPrivacyAggregator(BaseAggregator):
    """Agrégateur spécialisé pour la confidentialité différentielle"""

    def __init__(self, epsilon=DIFFERENTIAL_PRIVACY_EPSILON, delta=DIFFERENTIAL_PRIVACY_DELTA, clip_norm=1.0):
        super().__init__("DifferentialPrivacy")
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

    def aggregate(self, client_updates, client_weights, global_model):
        """Agrégation avec DP forte"""
        start_time = time.time()

        # Clipper toutes les mises à jour
        clipped_updates = [self._clip_update(update) for update in client_updates]

        # Moyenne pondérée
        aggregated_update = weighted_average(clipped_updates, client_weights)

        # Ajouter du bruit DP
        noisy_update = self._add_dp_noise(aggregated_update)

        # Appliquer au modèle
        global_weights = global_model.get_weights()
        new_global_weights = [
            global_w + update for global_w, update in zip(global_weights, noisy_update)
        ]

        # Métriques
        comm_cost = self.get_communication_cost(client_updates)
        agg_time = time.time() - start_time

        self.history['communication_costs'].append(comm_cost)
        self.history['aggregation_times'].append(agg_time)

        return new_global_weights

    def _clip_update(self, client_update):
        """Clipe une mise à jour client"""
        total_norm = sum(np.linalg.norm(update) ** 2 for update in client_update) ** 0.5

        if total_norm > self.clip_norm:
            return [update * self.clip_norm / total_norm for update in client_update]
        return client_update

    def _add_dp_noise(self, aggregated_update):
        """Ajoute du bruit différentiellement privé"""
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * self.clip_norm / self.epsilon

        noisy_update = []
        for update in aggregated_update:
            noise = np.random.normal(0, sigma, update.shape)
            noisy_update.append(update + noise)

        return noisy_update

    def prepare_client_update(self, client_id, local_model, global_model):
        """Prépare une mise à jour clippée"""
        local_weights = local_model.get_weights()
        global_weights = global_model.get_weights()

        client_update = [local_w - global_w for local_w, global_w in zip(local_weights, global_weights)]
        return self._clip_update(client_update)