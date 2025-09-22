"""
Serveur MPC avec protocole inter-clients
"""
import time
from federated_base import BaseFederatedServer
from mpc.secure_aggregation import MPCAggregator
import numpy as np


class MPCServer(BaseFederatedServer):
    """Serveur MPC avec gestion du protocole complet"""

    def __init__(self, model_type="standard"):
        super().__init__(MPCAggregator(), model_type)
        self.name = "MPC-InterClientServer"
        self.total_communication_cost = 0

    def train_round(self, clients, test_data, selection_ratio=1.0):
        """Round d'entraînement avec protocole MPC complet"""
        round_start_time = time.time()

        selected_clients = self.select_clients(clients, selection_ratio)

        if len(selected_clients) < 3:  # MPC_THRESHOLD
            print(f"Pas assez de clients pour MPC: {len(selected_clients)}")
            return self.evaluate(test_data)

        print(f"Round {self.current_round}: Protocole MPC avec {len(selected_clients)} clients")

        # ÉTAPE 1: Entraînement local standard
        print("  -> Entraînement local...")
        for client in selected_clients:
            client.update_model(self.global_model)
            client.train_local()

        # ÉTAPE 2: Création des shares après entraînement
        print("  -> Création des shares...")
        shares_creation_time = time.time()
        for client in selected_clients:
            client.create_shares_after_training()
        shares_time = time.time() - shares_creation_time

        # ÉTAPE 3: Communication inter-clients des shares
        print("  -> Échange des shares entre clients...")
        inter_client_comm_time = time.time()
        total_inter_client_cost = 0

        for client in selected_clients:
            comm_cost = client.send_shares_to_clients(selected_clients)
            total_inter_client_cost += comm_cost

        inter_comm_time = time.time() - inter_client_comm_time

        # ÉTAPE 4: Reconstruction des modèles locaux
        print("  -> Reconstruction des modèles...")
        reconstruction_time = time.time()
        client_updates = []
        client_weights = []

        for client in selected_clients:
            try:
                reconstructed_weights = client.reconstruct_local_model_from_shares()

                # VALIDATION AJOUTÉE:
                if self._validate_reconstructed_weights(reconstructed_weights):
                    # Calculer l'update seulement si poids valides
                    local_weights = client.local_model.get_weights()
                    global_weights = self.global_model.get_weights()
                    update = [local_w - global_w for local_w, global_w in
                              zip(local_weights, global_weights)]

                    client_updates.append(update)
                    client_weights.append(client.get_data_size())
                else:
                    print(f"Client {client.client_id}: Poids reconstruits invalides - ignoré")

            except Exception as e:
                print(f"Erreur reconstruction client {client.client_id}: {e}")
                continue
        recon_time = time.time() - reconstruction_time

        if not client_updates:
            print("Aucune reconstruction réussie")
            return self.evaluate(test_data)



        # ÉTAPE 5: Agrégation standard au serveur
        print("  -> Agrégation finale...")
        try:
            new_weights = self.aggregator.aggregate(
                client_updates, client_weights, self.global_model
            )
            self.global_model.set_weights(new_weights)
        except Exception as e:
            print(f"Erreur agrégation: {e}")
            return self.evaluate(test_data)

        # ÉTAPE 6: Évaluation et métriques
        test_acc = self.evaluate(test_data)
        round_time = time.time() - round_start_time

        # Calculer coûts de communication
        client_to_server_cost = 0
        for update in client_updates:
            for layer in update:
                try:
                    client_to_server_cost += int(np.prod(layer.shape))
                except:
                    client_to_server_cost += 1000  # Valeur par défaut

        total_comm_cost = total_inter_client_cost + client_to_server_cost
        self.total_communication_cost += total_comm_cost

        # Métriques détaillées
        self.metrics['test_accuracy'].append(test_acc)
        self.metrics['round_times'].append(round_time)
        self.metrics['inter_client_comm_cost'] = getattr(self.metrics, 'inter_client_comm_cost', [])
        self.metrics['inter_client_comm_cost'].append(total_inter_client_cost)
        self.metrics['total_comm_cost'] = getattr(self.metrics, 'total_comm_cost', [])
        self.metrics['total_comm_cost'].append(total_comm_cost)

        print(f"  -> Temps: {round_time:.2f}s | Comm inter-clients: {total_inter_client_cost} | Accuracy: {test_acc:.4f}")

        self.current_round += 1
        return test_acc

    def evaluate(self, test_data):
        """Évaluation standard"""
        x_test, y_test = test_data
        loss, accuracy = self.global_model.evaluate(x_test, y_test, verbose=0)
        return accuracy

    def get_communication_stats(self):
        """Statistiques de communication"""
        return {
            'total_communication_cost': self.total_communication_cost,
            'inter_client_costs': self.metrics.get('inter_client_comm_cost', []),
            'total_costs_per_round': self.metrics.get('total_comm_cost', [])
        }

    def _validate_reconstructed_weights(self, weights):
        """Valide les poids reconstruits"""
        if not weights:
            return False

        try:
            for layer_weights in weights:
                if np.any(np.isnan(layer_weights)) or np.any(np.isinf(layer_weights)):
                    return False
                if np.max(np.abs(layer_weights)) > 1000:
                    return False
            return True
        except Exception:
            return False