"""
Serveur fédéré
"""
import numpy as np
import time
import json
from datetime import datetime
from models import create_cifar10_cnn, create_mnist_cnn
from config import COMMUNICATION_ROUNDS


class FederatedServer:
    """Serveur pour l'apprentissage fédéré"""

    def __init__(self, aggregator, model_type="standard", dataset="cifar10"):
        self.aggregator = aggregator
        self.model_type = model_type
        self.dataset = dataset
        if dataset == "cifar10":
            self.global_model = create_cifar10_cnn(model_type)
        elif dataset == "mnist":
            self.global_model = create_mnist_cnn(model_type)
        self.current_round = 0
        self.metrics = {
            'train_accuracy': [],
            'test_accuracy': [],
            'communication_costs': [],
            'aggregation_times': [],
            'participating_clients': [],
            'round_times': []
        }
        self.client_selection_strategy = "random"

    def select_clients(self, clients, selection_ratio=1.0):
        """Sélectionne les clients pour ce tour"""
        if self.client_selection_strategy == "random":
            num_selected = max(1, int(len(clients) * selection_ratio))
            return list(np.random.choice(clients, num_selected, replace=False))
        elif self.client_selection_strategy == "all":
            return clients
        elif self.client_selection_strategy == "available":
            # Sélectionner seulement les clients disponibles
            available = [c for c in clients if hasattr(c, 'should_participate') and c.should_participate()]
            return available if available else [clients[0]]  # Au moins un client

    def train_round(self, clients, test_data, selection_ratio=1.0):
        """Un tour d'entraînement fédéré"""
        round_start_time = time.time()

        # Sélection des clients
        selected_clients = self.select_clients(clients, selection_ratio)

        client_updates = []
        client_weights = []
        participating_ids = []

        print(f"Round {self.current_round}: {len(selected_clients)} clients sélectionnés")

        # Phase d'entraînement local
        for client in selected_clients:
            try:
                # Vérifier la disponibilité pour les clients avec contraintes
                if hasattr(client, 'should_participate') and not client.should_participate():
                    continue

                client.update_model(self.global_model)
                client.train_local()

                update = client.get_update(self.global_model)
                client_updates.append(update)
                client_weights.append(client.get_data_size())
                participating_ids.append(client.client_id)

            except Exception as e:
                print(f"Erreur avec client {client.client_id}: {e}")
                continue

        if not client_updates:
            print("Aucune mise à jour reçue - tour ignoré")
            self.metrics['test_accuracy'].append(
                self.metrics['test_accuracy'][-1] if self.metrics['test_accuracy'] else 0.0)
            return self.evaluate(test_data)

        # Agrégation
        try:
            new_weights = self.aggregator.aggregate(
                client_updates, client_weights, self.global_model
            )
            self.global_model.set_weights(new_weights)
        except Exception as e:
            print(f"Erreur d'agrégation: {e}")
            return self.evaluate(test_data)

        # Évaluation
        test_acc = self.evaluate(test_data)

        # Métriques
        round_time = time.time() - round_start_time
        self.metrics['test_accuracy'].append(test_acc)
        self.metrics['participating_clients'].append(participating_ids)
        self.metrics['round_times'].append(round_time)

        if hasattr(self.aggregator, 'history'):
            if self.aggregator.history['communication_costs']:
                self.metrics['communication_costs'].append(
                    self.aggregator.history['communication_costs'][-1]
                )
            if self.aggregator.history['aggregation_times']:
                self.metrics['aggregation_times'].append(
                    self.aggregator.history['aggregation_times'][-1]
                )

        self.aggregator.update_round()
        self.current_round += 1

        return test_acc

    def evaluate(self, test_data):
        """Évalue le modèle global"""
        x_test, y_test = test_data
        loss, accuracy = self.global_model.evaluate(x_test, y_test, verbose=0)
        return accuracy

    def train_federated(self, clients, test_data, rounds=COMMUNICATION_ROUNDS, selection_ratio=1.0):
        """Entraînement fédéré complet"""
        print(f"Démarrage entraînement fédéré avec {self.aggregator.name}")
        print(f"Clients: {len(clients)}, Rounds: {rounds}")

        start_time = time.time()

        for round_num in range(rounds):
            acc = self.train_round(clients, test_data, selection_ratio)

            if round_num % 10 == 0 or round_num == rounds - 1:
                print(f"Round {round_num}: Test Accuracy = {acc:.4f}")

        total_time = time.time() - start_time
        print(f"Entraînement terminé en {total_time:.2f}s")

        return self.global_model, self.metrics

    def save_results(self, filepath):
        """Sauvegarde les résultats"""
        results = {
            'aggregator': self.aggregator.name,
            'final_accuracy': self.metrics['test_accuracy'][-1] if self.metrics['test_accuracy'] else 0,
            'rounds_completed': self.current_round,
            'total_time': sum(self.metrics['round_times']),
            'metrics': self.metrics,
            'aggregator_stats': self.aggregator.get_stats() if hasattr(self.aggregator, 'get_stats') else {},
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

    def get_convergence_analysis(self):
        """Analyse de convergence"""
        if len(self.metrics['test_accuracy']) < 2:
            return {}

        accuracies = np.array(self.metrics['test_accuracy'])

        return {
            'final_accuracy': accuracies[-1],
            'max_accuracy': np.max(accuracies),
            'convergence_round': np.argmax(accuracies),
            'improvement_rate': np.mean(np.diff(accuracies[:min(20, len(accuracies))])),
            'stability': np.std(accuracies[-10:]) if len(accuracies) >= 10 else float('inf')
        }


class AdaptiveFederatedServer(FederatedServer):
    """Serveur avec adaptation dynamique"""

    def __init__(self, aggregator, model_type="standard"):
        super().__init__(aggregator, model_type)
        self.adaptive_selection = True
        self.performance_history = []

    def select_clients(self, clients, selection_ratio=1.0):
        """Sélection adaptative des clients"""
        if not self.adaptive_selection:
            return super().select_clients(clients, selection_ratio)

        # Stratégie adaptative basée sur les performances
        if len(self.performance_history) > 5:
            recent_perf = np.mean(self.performance_history[-5:])
            if recent_perf < 0.01:  # Convergence lente
                selection_ratio = min(1.0, selection_ratio * 1.2)  # Plus de clients
            elif recent_perf > 0.05:  # Convergence rapide
                selection_ratio = max(0.3, selection_ratio * 0.8)  # Moins de clients

        return super().select_clients(clients, selection_ratio)

    def train_round(self, clients, test_data, selection_ratio=1.0):
        """Tour avec adaptation"""
        prev_acc = self.metrics['test_accuracy'][-1] if self.metrics['test_accuracy'] else 0

        acc = super().train_round(clients, test_data, selection_ratio)

        # Enregistrer l'amélioration
        improvement = acc - prev_acc
        self.performance_history.append(improvement)

        return acc


class SecureFederatedServer(FederatedServer):
    """Serveur avec fonctionnalités de sécurité"""

    def __init__(self, aggregator, model_type="standard", min_clients=3):
        super().__init__(aggregator, model_type)
        self.min_clients = min_clients
        self.suspicious_clients = set()

    def validate_client_update(self, client_id, update):
        """Valide une mise à jour client"""
        # Vérifications de base
        if not update:
            return False

        # Vérifier la norme des mises à jour (détection d'attaques)
        total_norm = sum(np.linalg.norm(layer_update) for layer_update in update)
        if total_norm > 100:  # Seuil arbitraire
            self.suspicious_clients.add(client_id)
            print(f"Client {client_id} marqué comme suspect (norme: {total_norm:.2f})")
            return False

        return True

    def train_round(self, clients, test_data, selection_ratio=1.0):
        """Tour avec validation de sécurité"""
        # Filtrer les clients suspects
        safe_clients = [c for c in clients if c.client_id not in self.suspicious_clients]

        if len(safe_clients) < self.min_clients:
            print(f"Pas assez de clients sûrs ({len(safe_clients)} < {self.min_clients})")
            return self.evaluate(test_data)

        return super().train_round(safe_clients, test_data, selection_ratio)