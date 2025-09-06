"""
Comparaison des différentes méthodes d'agrégation
"""
from aggregations import FedAvgAggregator, FedProxAggregator, ScaffoldAggregator
from client import FederatedClient
from server import FederatedServer
import numpy as np

def compare_aggregation_methods(fed_data, test_data):
    methods = {
        'FedAvg': FedAvgAggregator(),
        'FedProx': FedProxAggregator(mu=0.01),
        'SCAFFOLD': ScaffoldAggregator(),
    }

    results = {}

    for method_name, aggregator in methods.items():
        print(f"\n=== Testing {method_name} ===")

        model, metrics = train_federated_with_aggregator(
            aggregator, fed_data, test_data
        )

        results[method_name] = {
            'final_accuracy': metrics['test_accuracy'][-1],
            'convergence_rate': calculate_convergence_rate(metrics),
            'communication_cost': calculate_communication_cost(metrics)
        }

    return results

def calculate_convergence_rate(metrics):
    """Calcule le taux de convergence"""
    accuracies = metrics['test_accuracy']
    return np.mean(np.diff(accuracies[:20])) if len(accuracies) > 20 else 0


def calculate_communication_cost(metrics):
    """Calcule le coût de communication"""
    return sum(metrics.get('communication_costs', []))


def compare_aggregation_methods(fed_data, test_data, model_type="standard"):
    """Comparaison complète"""
    methods = {
        'FedAvg': FedAvgAggregator(),
        'FedProx': FedProxAggregator(),
        'SCAFFOLD': ScaffoldAggregator(),
    }

    results = {}

    for method_name, aggregator in methods.items():
        print(f"\n=== Testing {method_name} ===")

        clients = [FederatedClient(i, data, aggregator)
                   for i, data in enumerate(fed_data)]
        server = FederatedServer(aggregator, model_type)

        model, metrics = server.train_federated(clients, test_data)

        results[method_name] = {
            'final_accuracy': metrics['test_accuracy'][-1],
            'convergence_rate': calculate_convergence_rate(metrics),
            'communication_cost': calculate_communication_cost(metrics)
        }

    return results