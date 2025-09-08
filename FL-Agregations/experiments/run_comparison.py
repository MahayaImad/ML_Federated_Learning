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
    """Calcule le taux de convergence avec vérification"""
    accuracies = metrics.get('test_accuracy', [])
    if len(accuracies) > 20:
        return np.mean(np.diff(accuracies[:20]))
    elif len(accuracies) > 1:
        return np.mean(np.diff(accuracies))
    else:
        return 0.0


def calculate_communication_cost(metrics):
    """Calcule le coût de communication avec vérification"""
    costs = metrics.get('communication_costs', [])
    return sum(costs) if costs else 0.0


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

        final_accuracy = metrics['test_accuracy'][-1] if metrics['test_accuracy'] else 0.0

        results[method_name] = {
            'final_accuracy': final_accuracy,
            'convergence_rate': calculate_convergence_rate(metrics),
            'communication_cost': calculate_communication_cost(metrics)
        }
    return results