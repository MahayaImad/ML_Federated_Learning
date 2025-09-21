"""
Agrégation sécurisée avec MPC
"""
import numpy as np
import time
from .secret_sharing import ShamirSecretSharing
from config import MPC_THRESHOLD


class MPCAggregator:
    """Agrégateur utilisant MPC"""

    def __init__(self):
        self.name = "MPC-SecureAgg"
        self.secret_sharing = ShamirSecretSharing(MPC_THRESHOLD)
        self.round_number = 0
        self.history = {
            'communication_costs': [],
            'aggregation_times': [],
            'mpc_overhead': []
        }

    def secure_aggregate(self, client_updates, client_weights, global_model):
        """Agrégation sécurisée avec MPC"""
        start_time = time.time()

        num_clients = len(client_updates)
        if num_clients < MPC_THRESHOLD:
            raise ValueError(f"Besoin d'au moins {MPC_THRESHOLD} clients")

        # 1. Chaque client partage ses mises à jour
        shared_updates = []
        for client_update in client_updates:
            client_shares = []
            for layer_update in client_update:
                layer_shares = self._share_layer(layer_update, num_clients)
                client_shares.append(layer_shares)
            shared_updates.append(client_shares)

        # 2. Calcul sécurisé de la moyenne pondérée
        aggregated_shares = self._compute_weighted_average_shares(
            shared_updates, client_weights, num_clients
        )

        # 3. Reconstruction du résultat
        aggregated_update = self._reconstruct_aggregation(aggregated_shares)

        # 4. Mise à jour du modèle global
        global_weights = global_model.get_weights()
        new_global_weights = [
            global_w + update for global_w, update in zip(global_weights, aggregated_update)
        ]
        # 5. Validation des poids reconstruits
        self._validate_reconstructed_weights(new_global_weights)

        # Métriques
        agg_time = time.time() - start_time
        mpc_overhead = self._calculate_mpc_overhead(client_updates, num_clients)

        self.history['aggregation_times'].append(agg_time)
        self.history['mpc_overhead'].append(mpc_overhead)

        return new_global_weights

    def _share_layer(self, layer_weights, num_clients):
        """Partage les poids d'une couche"""
        flat_weights = layer_weights.flatten()
        shares_matrix = []

        for weight in flat_weights:
            shares = self.secret_sharing.share_secret(weight, num_clients)
            shares_matrix.append(shares)

        return shares_matrix, layer_weights.shape

    def _compute_weighted_average_shares(self, shared_updates, weights, num_clients):
        """Calcule la moyenne pondérée sur les parts partagées"""
        # Normaliser les poids
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Pour chaque couche
        aggregated_shares = []
        num_layers = len(shared_updates[0])

        for layer_idx in range(num_layers):
            layer_shares = []
            shares_matrix, shape = shared_updates[0][layer_idx]

            # Pour chaque poids dans la couche
            for weight_idx in range(len(shares_matrix)):
                # Collecter toutes les parts pour ce poids
                weight_shares = []
                for client_idx in range(len(shared_updates)):
                    client_shares, _ = shared_updates[client_idx][layer_idx]
                    weight_shares.append(client_shares[weight_idx])

                # Calcul de la moyenne pondérée des parts
                averaged_shares = self._weighted_average_shares(
                    weight_shares, normalized_weights
                )
                layer_shares.append(averaged_shares)

            aggregated_shares.append((layer_shares, shape))

        return aggregated_shares

    def _weighted_average_shares(self, weight_shares, normalized_weights):
        """Moyenne pondérée des parts secrètes - VERSION SIMPLIFIÉE"""
        # Pour le contexte académique : approximation directe
        num_shares = len(weight_shares[0])
        result_shares = []

        for share_idx in range(num_shares):
            # Moyenne simple des valeurs des parts
            share_values = [shares[share_idx][1] for shares in weight_shares]
            avg_value = sum(v * w for v, w in zip(share_values, normalized_weights))

            result_shares.append((share_idx + 1, int(avg_value)))

        return result_shares

    def _reconstruct_aggregation(self, aggregated_shares):
        """Reconstruit le résultat final"""
        reconstructed_layers = []

        for layer_shares, shape in aggregated_shares:
            flat_weights = []

            for weight_shares in layer_shares:
                reconstructed_weight = self.secret_sharing.reconstruct_secret(weight_shares)
                flat_weights.append(reconstructed_weight)

            layer_weights = np.array(flat_weights).reshape(shape)
            reconstructed_layers.append(layer_weights)
            
        return reconstructed_layers

    def _calculate_mpc_overhead(self, client_updates, num_clients):
        """Calcule l'overhead MPC"""
        # Simplification : facteur multiplicatif basé sur le nombre de parts
        base_size = sum(np.prod(update.shape) for client_update in client_updates
                        for update in client_update)
        return base_size * num_clients * MPC_THRESHOLD

    def _validate_reconstructed_weights(self, weights):
        """Valide que les poids reconstruits sont cohérents"""
        for layer_weights in weights:
            if np.any(np.isnan(layer_weights)) or np.any(np.isinf(layer_weights)):
                raise ValueError("Poids reconstruits invalides (NaN/Inf)")
            if np.max(np.abs(layer_weights)) > 1000:  # Seuil arbitraire
                raise ValueError("Poids reconstruits trop grands")