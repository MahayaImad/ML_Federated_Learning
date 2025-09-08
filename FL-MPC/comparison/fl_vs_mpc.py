"""
Comparaison FL classique vs FL-MPC
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# Importer FL classique
sys.path.append('../FL-Agregations')
from aggregations import FedAvgAggregator
from client import FederatedClient as ClassicClient
from server import FederatedServer as ClassicServer

# Importer FL-MPC
from client import MPCClient
from server import MPCServer


def compare_fl_vs_mpc(fed_data, test_data, rounds=30):
    """Compare FL classique avec FL-MPC"""

    print("=== COMPARAISON FL CLASSIQUE vs FL-MPC ===")

    results = {}

    # 1. FL Classique
    print("\n--- FL Classique (FedAvg) ---")
    classic_clients = [ClassicClient(i, data, FedAvgAggregator())
                       for i, data in enumerate(fed_data)]
    classic_server = ClassicServer(FedAvgAggregator())

    _, classic_metrics = classic_server.train_federated(
        classic_clients, test_data, rounds
    )

    results['classic'] = {
        'accuracy': classic_metrics['test_accuracy'],
        'times': classic_metrics['round_times'],
        'communication': classic_metrics.get('communication_costs', [])
    }

    # 2. FL-MPC
    print("\n--- FL avec MPC ---")
    mpc_clients = [MPCClient(i, data) for i, data in enumerate(fed_data)]
    mpc_server = MPCServer()

    try:
        _, mpc_metrics = mpc_server.train_federated(
            mpc_clients, test_data, rounds
        )

        results['mpc'] = {
            'accuracy': mpc_metrics['test_accuracy'],
            'times': mpc_metrics['round_times'],
            'mpc_overhead': mpc_server.aggregator.history['mpc_overhead']
        }
    except Exception as e:
        print(f"Erreur FL-MPC: {e}")
        results['mpc'] = None

    return results


def plot_comparison(results):
    """Visualise la comparaison"""
    if results['mpc'] is None:
        print("Impossible de tracer - erreur MPC")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy
    axes[0].plot(results['classic']['accuracy'], label='FL Classique', marker='o')
    axes[0].plot(results['mpc']['accuracy'], label='FL-MPC', marker='s')
    axes[0].set_title('Précision au fil des tours')
    axes[0].set_xlabel('Tours')
    axes[0].set_ylabel('Précision')
    axes[0].legend()
    axes[0].grid(True)

    # Temps par tour
    axes[1].plot(results['classic']['times'], label='FL Classique', marker='o')
    axes[1].plot(results['mpc']['times'], label='FL-MPC', marker='s')
    axes[1].set_title('Temps par tour')
    axes[1].set_xlabel('Tours')
    axes[1].set_ylabel('Temps (s)')
    axes[1].legend()
    axes[1].grid(True)

    # Overhead MPC
    if results['mpc']['mpc_overhead']:
        axes[2].plot(results['mpc']['mpc_overhead'], label='Overhead MPC', marker='^', color='red')
        axes[2].set_title('Overhead de communication MPC')
        axes[2].set_xlabel('Tours')
        axes[2].set_ylabel('Overhead')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Statistiques finales
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"FL Classique - Précision finale: {results['classic']['accuracy'][-1]:.4f}")
    print(f"FL-MPC - Précision finale: {results['mpc']['accuracy'][-1]:.4f}")
    print(f"FL Classique - Temps moyen/tour: {np.mean(results['classic']['times']):.2f}s")
    print(f"FL-MPC - Temps moyen/tour: {np.mean(results['mpc']['times']):.2f}s")