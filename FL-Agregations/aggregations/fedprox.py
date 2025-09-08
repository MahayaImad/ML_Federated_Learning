"""
Implémentation de l'algorithme FedProx
"""
import time
import numpy as np
import tensorflow as tf
from .base_aggregator import BaseAggregator, weighted_average, subtract_weights
from config import FEDPROX_MU
from config import BATCH_SIZE

class FedProxAggregator(BaseAggregator):
    """Agrégateur FedProx avec terme de régularisation proximale"""

    def __init__(self, mu=FEDPROX_MU):
        super().__init__("FedProx")
        self.mu = mu  # Paramètre de régularisation proximale

    def aggregate(self, client_updates, client_weights, global_model):
        """
        Agrégation FedProx (identique à FedAvg au niveau serveur)
        La différence est dans l'entraînement local avec régularisation
        """
        start_time = time.time()

        # Moyenne pondérée standard
        aggregated_update = weighted_average(client_updates, client_weights)

        # Application au modèle global
        global_weights = global_model.get_weights()
        new_global_weights = [
            global_w + update for global_w, update in zip(global_weights, aggregated_update)
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
        """Même que FedAvg - la différence est dans l'entraînement local"""
        local_weights = local_model.get_weights()
        global_weights = global_model.get_weights()

        client_update = subtract_weights(local_weights, global_weights)
        return client_update

    def train_client_with_proximal_term(self, client_model, global_model, x_train, y_train, epochs):
        """
        Entraîne un client avec le terme de régularisation proximale

        Args:
            client_model: Modèle du client
            global_model: Modèle global (pour la régularisation)
            x_train, y_train: Données d'entraînement du client
            epochs: Nombre d'époques locales

        Returns:
            trained_model: Modèle entraîné
        """
        global_weights = global_model.get_weights()

        # Fonction de perte personnalisée avec terme proximal
        def proximal_loss(y_true, y_pred):
            # Perte standard
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

            # Terme de régularisation proximale
            proximal_term = 0.0
            current_weights = client_model.get_weights()

            for current_w, global_w in zip(current_weights, global_weights):
                diff = current_w - global_w
                proximal_term += tf.reduce_sum(tf.square(diff))

            proximal_term *= self.mu / 2.0

            return ce_loss + proximal_term

        # Compiler avec la nouvelle fonction de perte
        client_model.compile(
            optimizer=client_model.optimizer,
            loss=proximal_loss,
            metrics=['accuracy']
        )

        # Entraînement
        client_model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            verbose=0
        )

        return client_model

class AdaptiveFedProxAggregator(FedProxAggregator):
    """FedProx avec adaptation automatique du paramètre mu"""

    def __init__(self, initial_mu=FEDPROX_MU, adaptation_rate=0.1):
        super().__init__(initial_mu)
        self.name = "AdaptiveFedProx"
        self.initial_mu = initial_mu
        self.adaptation_rate = adaptation_rate
        self.client_divergences = []

    def aggregate(self, client_updates, client_weights, global_model):
        """Agrégation avec adaptation de mu"""
        # Calculer la divergence des clients
        update_norms = []
        for update in client_updates:
            norm = sum(np.linalg.norm(layer_update) for layer_update in update)
            update_norms.append(norm)

        avg_divergence = np.mean(update_norms)
        self.client_divergences.append(avg_divergence)

        # Adapter mu basé sur la divergence
        if len(self.client_divergences) > 1:
            if avg_divergence > np.mean(self.client_divergences[-5:]):
                # Augmenter mu si divergence croissante
                self.mu = min(self.mu * (1 + self.adaptation_rate), self.initial_mu * 3)
            else:
                # Diminuer mu si convergence
                self.mu = max(self.mu * (1 - self.adaptation_rate), self.initial_mu * 0.1)