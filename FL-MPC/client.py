"""
Client MPC avec communication inter-clients
"""
import numpy as np
from federated_base import BaseFederatedClient
from mpc.secret_sharing import ShamirSecretSharing
from config import MPC_THRESHOLD, LEARNING_RATE
import tensorflow as tf


class MPCClient(BaseFederatedClient):
    """Client MPC avec partage de secrets réel"""

    def __init__(self, client_id, data, total_clients):
        super().__init__(client_id, data)
        self.total_clients = total_clients
        self.secret_sharing = ShamirSecretSharing(MPC_THRESHOLD)
        self.received_shares = {}  # Shares reçues des autres clients
        self.my_shares = {}        # Mes shares à envoyer
        self.communication_cost = 0
        self.is_byzantine = False

    def create_shares_after_training(self):
        """Crée les shares après l'entraînement local"""
        local_weights = self.local_model.get_weights()
        self.my_shares = {}

        for layer_idx, layer_weights in enumerate(local_weights):
            layer_shares = []
            flat_weights = layer_weights.flatten()

            for weight_idx, weight in enumerate(flat_weights):
                # Créer shares pour ce poids
                shares = self.secret_sharing.share_secret(float(weight), self.total_clients)
                layer_shares.append(shares)

            self.my_shares[layer_idx] = {
                'shares': layer_shares,
                'shape': layer_weights.shape
            }

        return self.my_shares

    def send_shares_to_clients(self, other_clients):
        """Envoie mes shares aux autres clients"""
        communication_cost = 0

        for other_client in other_clients:
            if other_client.client_id != self.client_id:
                # Envoyer mes shares à ce client
                shares_for_client = self._extract_shares_for_client(other_client.client_id)
                other_client.receive_shares_from_client(self.client_id, shares_for_client)

                # Compter le coût de communication
                communication_cost += self._calculate_shares_size(shares_for_client)

        self.communication_cost += communication_cost
        return communication_cost

    def receive_shares_from_client(self, sender_id, shares_data):
        """Reçoit les shares d'un autre client"""
        self.received_shares[sender_id] = shares_data

    def _extract_shares_for_client(self, target_client_id):
        """Extrait les shares destinées à un client spécifique"""
        shares_for_client = {}

        for layer_idx, layer_data in self.my_shares.items():
            layer_shares_for_client = []

            for weight_shares in layer_data['shares']:
                client_share = None
                for point, value in weight_shares:
                    if point == target_client_id + 1:  # Les x commencent à 1
                        client_share = (point, value)
                        break

                if client_share is None:
                    raise ValueError(f"Share manquante pour client {target_client_id}")

                layer_shares_for_client.append(client_share)

            shares_for_client[layer_idx] = {
                'shares': layer_shares_for_client,
                'shape': layer_data['shape']
            }
        return shares_for_client

    def reconstruct_local_model_from_shares(self):
        """Reconstruit le modèle local à partir des shares reçues"""
        if len(self.received_shares) < MPC_THRESHOLD - 1:
            raise ValueError(f"Pas assez de shares reçues: {len(self.received_shares)}")

        reconstructed_weights = []

        for layer_idx in self.my_shares.keys():
            all_shares_for_layer = []
            layer_shape = self.my_shares[layer_idx]['shape']
            num_weights = len(self.my_shares[layer_idx]['shares'])

            for weight_idx in range(num_weights):
                weight_shares = []

                # CORRECTION: Ma propre share
                my_shares_list = self.my_shares[layer_idx]['shares'][weight_idx]
                my_share = None
                for point, value in my_shares_list:
                    if point == self.client_id + 1:
                        my_share = (point, value)
                        break

                if my_share is not None:
                    weight_shares.append(my_share)

                # Shares des autres clients
                for sender_id, shares_data in self.received_shares.items():
                    if layer_idx in shares_data and weight_idx < len(shares_data[layer_idx]['shares']):
                        other_share = shares_data[layer_idx]['shares'][weight_idx]
                        weight_shares.append(other_share)

                # Reconstruire ce poids
                if len(weight_shares) >= MPC_THRESHOLD:
                    reconstructed_weight = self.secret_sharing.reconstruct_secret(weight_shares)
                    all_shares_for_layer.append(reconstructed_weight)
                else:
                    # Fallback : utiliser le poids original
                    original_weight = self.local_model.get_weights()[layer_idx].flatten()[weight_idx]
                    all_shares_for_layer.append(float(original_weight))

            # Reformater en shape originale
            layer_weights = np.array(all_shares_for_layer).reshape(layer_shape)
            reconstructed_weights.append(layer_weights)

        return reconstructed_weights

    def _calculate_shares_size(self, shares_data):
        """Calcule la taille des shares envoyées"""
        total_size = 0
        for layer_data in shares_data.values():
            total_size += len(layer_data['shares']) * 2  # (point, valeur) par share
        return total_size

    def get_total_communication_cost(self):
        """Retourne le coût total de communication"""
        return self.communication_cost

    def train_local(self, epochs=3):
        """Entraînement local MPC"""
        if self.local_model is None:
            raise ValueError("Modèle local non initialisé")

        # CORRECTION : Recompiler le modèle
        self.local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',  # ←← CHANGEMENT ICI
            metrics=['accuracy']
        )

        # Entraînement
        history = self.local_model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        return self.local_model