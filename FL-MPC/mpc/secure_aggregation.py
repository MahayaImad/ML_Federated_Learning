"""
Agrégation sécurisée avec MPC - VERSION INTER-CLIENTS
Pas d'agrégation MPC côté serveur, juste FedAvg classique
Le MPC se fait entre clients avant envoi au serveur
"""
import numpy as np
import time
from config import MPC_THRESHOLD


class MPCAggregator:
    """
    Agrégateur simplifié pour serveur MPC
    Le vrai MPC se fait entre clients, pas au serveur
    """

    def __init__(self):
        self.name = "MPC-InterClientAgg"
        self.threshold = MPC_THRESHOLD
        self.history = {
            'aggregation_times': [],
            'communication_costs': [],
            'mpc_overhead': []
        }

    def aggregate(self, client_updates, client_weights, global_model):
        """
        Agrégation standard FedAvg au serveur
        Les clients ont déjà fait le MPC entre eux
        """
        start_time = time.time()

        if len(client_updates) < self.threshold:
            raise ValueError(f"Besoin d'au moins {self.threshold} clients")

        # FedAvg classique : moyenne pondérée des updates
        aggregated_update = self._fedavg_aggregation(client_updates, client_weights)

        # Application de l'update au modèle global
        global_weights = global_model.get_weights()
        new_weights = []

        for global_layer, update_layer in zip(global_weights, aggregated_update):
            new_layer = global_layer + update_layer
            new_weights.append(new_layer)

        # Métriques
        agg_time = time.time() - start_time
        self.history['aggregation_times'].append(float(agg_time))

        return new_weights

    def _fedavg_aggregation(self, client_updates, client_weights):
        """FedAvg standard : moyenne pondérée"""
        total_weight = sum(client_weights)

        if total_weight == 0:
            raise ValueError("Poids total des clients est zéro")

        # Normaliser les poids
        normalized_weights = [w / total_weight for w in client_weights]

        # Initialiser l'agrégation
        aggregated_layers = []

        for layer_idx in range(len(client_updates[0])):
            # Moyenne pondérée pour cette couche
            layer_sum = np.zeros_like(client_updates[0][layer_idx])

            for client_idx, update in enumerate(client_updates):
                weight = normalized_weights[client_idx]
                layer_sum += update[layer_idx] * weight

            aggregated_layers.append(layer_sum)

        return aggregated_layers

    def get_communication_overhead(self):
        """Retourne l'overhead de communication estimé"""
        return {
            'aggregation_times': self.history['aggregation_times'],
            'total_overhead': sum(self.history.get('mpc_overhead', []))
        }


# Classe utilitaire pour les calculs MPC côté client
class ClientMPCHelper:
    """
    Classe helper pour les calculs MPC côté client
    Utilisée dans la reconstruction des modèles
    """

    @staticmethod
    def validate_shares_consistency(shares_dict, expected_clients):
        """Valide la cohérence des shares reçues"""
        received_clients = set(shares_dict.keys())

        if len(received_clients) < MPC_THRESHOLD - 1:
            raise ValueError(f"Pas assez de shares: {len(received_clients)} < {MPC_THRESHOLD - 1}")

        # Vérifier la structure des shares
        first_shares = next(iter(shares_dict.values()))
        expected_layers = set(first_shares.keys())

        for client_id, shares_data in shares_dict.items():
            if set(shares_data.keys()) != expected_layers:
                raise ValueError(f"Structure incohérente pour client {client_id}")

        return True

    @staticmethod
    def calculate_reconstruction_cost(shares_dict, layer_shapes):
        """Calcule le coût de reconstruction"""
        total_cost = 0

        for client_id, shares_data in shares_dict.items():
            for layer_idx, layer_data in shares_data.items():
                # Coût = nombre de shares * taille de chaque share
                num_shares = len(layer_data.get('shares', []))
                total_cost += num_shares * 2  # (point, valeur) par share

        return total_cost

    @staticmethod
    def estimate_communication_savings(num_clients, model_size):
        """
        Estime les économies de communication du MPC
        vs envoi direct des modèles
        """
        # Communication directe : chaque client envoie tout son modèle
        direct_comm = num_clients * model_size

        # Communication MPC : shares entre clients + modèles reconstruits
        mpc_inter_client = num_clients * (num_clients - 1) * model_size * 0.1  # shares plus petites
        mpc_to_server = num_clients * model_size  # modèles reconstruits
        mpc_total = mpc_inter_client + mpc_to_server

        return {
            'direct_communication': direct_comm,
            'mpc_communication': mpc_total,
            'overhead_ratio': mpc_total / direct_comm if direct_comm > 0 else float('inf'),
            'privacy_benefit': 'Modèles individuels protégés par secret sharing'
        }


# Classe pour les métriques MPC
class MPCMetrics:
    """Collecteur de métriques spécifiques au MPC"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Remet à zéro les métriques"""
        self.metrics = {
            'shares_creation_time': [],
            'inter_client_communication': [],
            'reconstruction_time': [],
            'total_mpc_time': [],
            'communication_costs': [],
            'accuracy_impact': []
        }

    def record_shares_creation(self, time_taken, num_clients):
        """Enregistre le temps de création des shares"""
        self.metrics['shares_creation_time'].append({
            'time': float(time_taken),
            'clients': int(num_clients)
        })

    def record_inter_client_comm(self, comm_cost, num_clients):
        """Enregistre la communication inter-clients"""
        self.metrics['inter_client_communication'].append({
            'cost': float(comm_cost),
            'clients': int(num_clients)
        })

    def record_reconstruction(self, time_taken, success_rate):
        """Enregistre les résultats de reconstruction"""
        self.metrics['reconstruction_time'].append({
            'time': float(time_taken),
            'success_rate': float(success_rate)
        })

    def get_summary(self):
        """Retourne un résumé des métriques"""
        summary = {}

        for metric_name, values in self.metrics.items():
            if values and isinstance(values[0], dict):
                # Métriques structurées
                times = [v.get('time', 0) for v in values]
                summary[metric_name] = {
                    'avg_time': np.mean(times) if times else 0,
                    'total_time': np.sum(times) if times else 0,
                    'count': len(values)
                }
            else:
                # Métriques simples
                summary[metric_name] = {
                    'values': values,
                    'avg': np.mean(values) if values else 0,
                    'total': np.sum(values) if values else 0
                }

        return summary

    def export_for_json(self):
        """Exporte les métriques dans un format JSON-compatible"""
        clean_metrics = {}

        for key, values in self.metrics.items():
            if isinstance(values, list):
                clean_values = []
                for v in values:
                    if isinstance(v, dict):
                        clean_v = {k: float(val) if isinstance(val, (np.integer, np.floating)) else val
                                  for k, val in v.items()}
                        clean_values.append(clean_v)
                    else:
                        clean_values.append(float(v) if isinstance(v, (np.integer, np.floating)) else v)
                clean_metrics[key] = clean_values
            else:
                clean_metrics[key] = values

        return clean_metrics