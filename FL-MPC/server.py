"""
Serveur MPC
"""
import time

from federated_base import BaseFederatedServer
from mpc.secure_aggregation import MPCAggregator


class MPCServer(BaseFederatedServer):
    """Serveur pour FL avec MPC"""

    def __init__(self, model_type="standard"):
        super().__init__(MPCAggregator(), model_type)
        self.name = "MPC-FederatedServer"

    def train_round(self, clients, test_data, selection_ratio=1.0):
        """Round d'entraînement avec MPC"""
        round_start_time = time.time()

        # Sélection des clients (besoin minimum pour MPC)
        selected_clients = self.select_clients(clients, selection_ratio)

        if len(selected_clients) < self.aggregator.secret_sharing.threshold:
            print(
                f"Pas assez de clients pour MPC ({len(selected_clients)} < {self.aggregator.secret_sharing.threshold})")
            return self.evaluate(test_data)

        # Phase d'entraînement local
        client_updates = []
        client_weights = []

        for client in selected_clients:
            client.update_model(self.global_model)
            client.train_local()

            # Utiliser l'update sécurisée
            update = client.get_secure_update(self.global_model)
            client_updates.append(update)
            client_weights.append(client.get_data_size())

        # Agrégation sécurisée avec MPC

        try:
            new_weights = self.aggregator.secure_aggregate(
                client_updates, client_weights, self.global_model
            )
            self.global_model.set_weights(new_weights)
        except Exception as e:
            print(f"Erreur MPC: {e}")
            # Garder l'accuracy précédente au lieu de 0.1
            if self.metrics['test_accuracy']:
                return self.metrics['test_accuracy'][-1]
            return 0.1

        # Évaluation et métriques
        test_acc = self.evaluate(test_data)

        round_time = time.time() - round_start_time
        self.metrics['test_accuracy'].append(test_acc)
        self.metrics['round_times'].append(round_time)

        self.current_round += 1
        return test_acc

    def evaluate(self, test_data):
        """Évalue le modèle global"""
        x_test, y_test = test_data
        loss, accuracy = self.global_model.evaluate(x_test, y_test, verbose=0)
        return accuracy