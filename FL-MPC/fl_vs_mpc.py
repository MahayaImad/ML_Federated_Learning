"""
Comparaison FL classique vs FL-MPC
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from config import RESULTS_DIR, PLOTS_DIR


from federated_base import BaseFederatedClient, BaseFederatedServer, SimpleFedAvgAggregator

# Importer FL-MPC
from client import MPCClient
from server import MPCServer


def compare_fl_vs_mpc(fed_data, test_data, rounds=30):
    """Compare FL classique avec FL-MPC"""

    print("=== COMPARAISON FL CLASSIQUE vs FL-MPC ===")

    results = {}



    # 2. FL-MPC
    print("\n--- FL avec MPC ---")
    # NOUVEAU CODE
    total_clients = len(fed_data)
    mpc_clients = [MPCClient(i, data, total_clients) for i, data in enumerate(fed_data)]
    mpc_server = MPCServer()

    try:
        _, mpc_metrics = mpc_server.train_federated(
            mpc_clients, test_data, rounds
        )

        results['mpc'] = {
            'accuracy': mpc_metrics['test_accuracy'],
            'times': mpc_metrics['round_times'],
            'inter_client_comm': mpc_metrics.get('inter_client_comm_cost', []),
            'total_comm': mpc_metrics.get('total_comm_cost', [])
        }
    except Exception as e:
        print(f"Erreur FL-MPC: {e}")
        results['mpc'] = None

        # 1. FL Classique
    print("\n--- FL Classique (FedAvg) ---")
    classic_clients = [BaseFederatedClient(i, data) for i, data in enumerate(fed_data)]
    classic_server = BaseFederatedServer(SimpleFedAvgAggregator())

    _, classic_metrics = classic_server.train_federated(
        classic_clients, test_data, rounds
    )

    results['classic'] = {
        'accuracy': classic_metrics['test_accuracy'],
        'times': classic_metrics['round_times'],
        'communication': classic_metrics.get('communication_costs', [])
    }

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
    axes[0].set_title('Accuracy au fil des rounds')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Temps par tour
    axes[1].plot(results['classic']['times'], label='FL Classique', marker='o')
    axes[1].plot(results['mpc']['times'], label='FL-MPC', marker='s')
    axes[1].set_title('Temps par round')
    axes[1].set_xlabel('Rounds')
    axes[1].set_ylabel('Temps (s)')
    axes[1].legend()
    axes[1].grid(True)

    # Overhead MPC
    if results['mpc']['mpc_overhead']:
        axes[2].plot(results['mpc']['mpc_overhead'], label='Overhead MPC', marker='^', color='red')
        axes[2].set_title('Overhead de communication MPC')
        axes[2].set_xlabel('Rounds')
        axes[2].set_ylabel('Overhead')
        axes[2].legend()
        axes[2].grid(True)

    plt.tight_layout()

    # Statistiques finales
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"FL Classique - Accuracy finale: {results['classic']['accuracy'][-1]:.4f}")
    print(f"FL-MPC - Accuracy finale: {results['mpc']['accuracy'][-1]:.4f}")
    print(f"FL Classique - Temps moyen/round: {np.mean(results['classic']['times']):.2f}s")
    print(f"FL-MPC - Temps moyen/round: {np.mean(results['mpc']['times']):.2f}s")

    # Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(PLOTS_DIR, f"comparison_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Graphique sauvegardé: {plot_file}")
    return plot_file

def save_comparison_results(results, args):
    """Sauvegarde les résultats de comparaison"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # Sauvegarde JSON
        results_data = {
            'timestamp': timestamp,
            'configuration': {
                'iid': args.iid,
            },
            'results': results
        }

        json_file = os.path.join(RESULTS_DIR, f"comparison_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"Résultats sauvegardés: {json_file}")

    except Exception as e:
        print(f"❌ Erreur sauvegarde JSON: {e}")