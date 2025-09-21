"""
Implémentation de l'algorithme MOON (Model-Contrastive Federated Learning)
"""
import time
import numpy as np
import tensorflow as tf
from .base_aggregator import BaseAggregator, weighted_average, subtract_weights

class MoonAggregator(BaseAggregator):
    """Agrégateur MOON avec apprentissage contrastif"""

    def __init__(self, temperature=0.5, mu=1.0):
        super().__init__("MOON")
        self.temperature = temperature
        self.mu = mu
        self.previous_global_model = None

    def prepare_client_update(self, client_id, local_model, global_model):
        """
        Prépare la mise à jour client pour MOON

        Args:
            client_id: ID du client
            local_model: modèle local du client
            global_model: modèle global actuel

        Returns:
            dict: mise à jour avec perte contrastive
        """
        # Obtenir les poids du modèle local
        local_weights = local_model.get_weights()
        global_weights = global_model.get_weights()

        # Calculer le delta (comme FedAvg)
        model_update = subtract_weights(local_weights, global_weights)

        # Calculer la perte contrastive si on a un modèle global précédent
        contrastive_loss = 0.0
        if self.previous_global_model is not None:
            contrastive_loss = self.get_contrastive_loss(
                local_weights,
                global_weights,
                self.previous_global_model
            )

        return {
            'model_update': model_update,
            'contrastive_loss': contrastive_loss,
            'client_id': client_id
        }

    def aggregate(self, client_updates, client_weights, global_model):
        """Agrégation MOON avec contrastive learning"""
        start_time = time.time()

        if not client_updates:
            return global_model.get_weights()

        # Sauvegarder le modèle global précédent
        if self.previous_global_model is None:
            self.previous_global_model = [np.copy(w) for w in global_model.get_weights()]

        # Extraire les mises à jour des modèles
        model_updates = []
        contrastive_losses = []

        for update in client_updates:
            if isinstance(update, dict) and 'model_update' in update:
                model_updates.append(update['model_update'])
                contrastive_losses.append(update.get('contrastive_loss', 0.0))
            else:
                # Format simple (pour compatibilité)
                model_updates.append(update)
                contrastive_losses.append(0.0)

        # Agrégation standard (comme FedAvg)
        aggregated_update = weighted_average(model_updates, client_weights)

        # Application au modèle global
        global_weights = global_model.get_weights()
        new_global_weights = [
            global_w + update for global_w, update in zip(global_weights, aggregated_update)
        ]

        # Mise à jour du modèle global précédent
        self.previous_global_model = [np.copy(w) for w in new_global_weights]

        # Métriques
        comm_cost = self.get_communication_cost(client_updates)
        agg_time = time.time() - start_time
        avg_contrastive_loss = np.mean(contrastive_losses) if contrastive_losses else 0.0

        # Gestion de l'historique avec limite
        max_history = 100
        if len(self.history['communication_costs']) >= max_history:
            self.history['communication_costs'].pop(0)
        if len(self.history['aggregation_times']) >= max_history:
            self.history['aggregation_times'].pop(0)

        self.history['communication_costs'].append(comm_cost)
        self.history['aggregation_times'].append(agg_time)

        # Ajouter métrique MOON
        if 'contrastive_losses' not in self.history:
            self.history['contrastive_losses'] = []
        self.history['contrastive_losses'].append(avg_contrastive_loss)

        return new_global_weights

    def get_contrastive_loss(self, local_model, global_model, previous_global):
        """Calcule la perte contrastive pour MOON"""
        try:
            # Représentation des modèles (utiliser les poids de la dernière couche)
            local_repr = local_model[-1].flatten()
            global_repr = global_model[-1].flatten()
            prev_repr = previous_global[-1].flatten()

            # Similarité cosinus
            def cosine_similarity(a, b):
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return np.dot(a, b) / (norm_a * norm_b)

            pos_sim = cosine_similarity(local_repr, global_repr) / self.temperature
            neg_sim = cosine_similarity(local_repr, prev_repr) / self.temperature

            # Contrastive loss avec stabilité numérique
            exp_pos = np.exp(np.clip(pos_sim, -10, 10))
            exp_neg = np.exp(np.clip(neg_sim, -10, 10))

            contrastive_loss = -np.log(exp_pos / (exp_pos + exp_neg + 1e-8) + 1e-8)

            return float(contrastive_loss)

        except Exception as e:
            print(f"Erreur dans le calcul de la perte contrastive: {e}")
            return 0.0