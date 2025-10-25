#!/usr/bin/env python3
"""
FL Comparison Framework
Compares: FedAvg, FedProx, IFCA, and Agglomerative Clustering
Datasets: MNIST, CIFAR10, MALNET
"""

import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

from config import *
from data_preparation import prepare_data
from models import initialize_global_model
from client import FederatedClient
from fedavg import train_fedavg
from fedprox import train_fedprox
from ifca import train_ifca
from agglomerative_inter import train_agglomerative_inter
from agglomerative_global import train_agglomerative_global
from utils import save_results, save_client_stats_csv, plot_comparison, create_summary_table

# Désactiver warnings TF
tf.get_logger().setLevel('ERROR')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='FL Comparison Framework')

    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar10', 'malnet'],
                        help='Dataset to use')

    parser.add_argument('--method', type=str, required=True,
                        choices=['fedavg', 'fedprox', 'ifca', 'agglomerative-inter', 'agglomerative-global', 'all'],
                        help='FL method to use (or "all" to compare all methods)')

    # Optional arguments
    parser.add_argument('--clients', type=int, default=CLIENTS,
                        help=f'Number of clients (default: {CLIENTS})')

    parser.add_argument('--rounds', type=int, default=COMMUNICATION_ROUNDS,
                        help=f'Communication rounds (default: {COMMUNICATION_ROUNDS})')

    parser.add_argument('--local-epochs', type=int, default=LOCAL_EPOCHS,
                        help=f'Local training epochs (default: {LOCAL_EPOCHS})')

    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')

    # Agglomerative specific
    parser.add_argument('--js-threshold', type=float, default=0.5,
                        help='JS divergence threshold for clustering (default: 0.5)')

    parser.add_argument('--selection-ratio', type=float, default=0.3,
                        help='Client selection ratio per cluster (default: 0.3)')

    parser.add_argument('--client-selection', type=str, default='size',
                        choices=['size', 'diversity', 'balanced', 'random'],
                        help='Client selection strategy: size (most data), diversity (most diverse), balanced (both), random')

    parser.add_argument('--selection-alpha', type=float, default=0.5,
                        help='Balance between size and diversity for balanced mode (0=size only, 1=diversity only, default: 0.5)')

    # FedProx specific
    parser.add_argument('--mu', type=float, default=0.01,
                        help='FedProx proximal term (default: 0.01)')

    # IFCA specific
    parser.add_argument('--num-clusters', type=int, default=5,
                        help='Number of clusters for IFCA (default: 5)')

    # General options
    parser.add_argument('--iid', action='store_true',
                        help='Use IID data distribution (default: non-IID)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use (-1 for CPU, default: 0)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def setup_gpu(gpu_id):
    """Configure GPU settings"""
    if gpu_id == -1:
        tf.config.set_visible_devices([], 'GPU')
        print("Using CPU")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and gpu_id < len(gpus):
            try:
                tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                print(f"Using GPU {gpu_id}: {gpus[gpu_id].name}")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print(f"GPU {gpu_id} not found, using CPU")


def run_method(method_name, clients, test_data, args):
    """Run a single FL method"""
    print(f"\n{'=' * 60}")
    print(f"Training: {method_name.upper()}")
    print(f"{'=' * 60}")

    if method_name == 'fedavg':
        results = train_fedavg(clients, test_data, args)
    elif method_name == 'fedprox':
        results = train_fedprox(clients, test_data, args)
    elif method_name == 'ifca':
        results = train_ifca(clients, test_data, args)
    elif method_name == 'agglomerative-inter':
        # Extract client data for clustering
        clients_data = [(c.x_train, c.y_train) for c in clients]
        results = train_agglomerative_inter(clients, clients_data, test_data, args)
    elif method_name == 'agglomerative-global':
        # Extract client data for clustering
        clients_data = [(c.x_train, c.y_train) for c in clients]
        results = train_agglomerative_global(clients, clients_data, test_data, args)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    results['method'] = method_name
    return results


def main():
    """Main execution function"""
    args = parse_arguments()

    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Setup GPU
    setup_gpu(args.gpu)

    # Print configuration
    print("\n" + "=" * 60)
    print("FL COMPARISON FRAMEWORK")
    print("=" * 60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Method: {args.method.upper()}")
    print(f"Clients: {args.clients}")
    print(f"Rounds: {args.rounds}")
    print(f"Local Epochs: {args.local_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Data Distribution: {'IID' if args.iid else 'Non-IID'}")
    if args.method == 'agglomerative-inter' or args.method == 'agglomerative-global':
        print(f"JS Threshold: {args.js_threshold}")
        print(f"Selection Ratio: {args.selection_ratio}")
        print(f"Client Selection: {args.client_selection}")
        if args.client_selection == 'balanced':
            print(f"Selection Alpha: {args.selection_alpha}")
    if args.method == 'fedprox':
        print(f"Mu (proximal term): {args.mu}")
    if args.method == 'ifca':
        print(f"Number of Clusters: {args.num_clusters}")
    print("=" * 60 + "\n")

    # Prepare data
    print("Preparing data...")
    fed_data, test_data, client_info = prepare_data(
        args.dataset, args.clients, args.iid
    )

    # Create clients
    clients = [
        FederatedClient(i, data, args.batch_size, args.dataset)
        for i, data in enumerate(fed_data)
    ]

    print(f"Setup complete: {len(clients)} clients, {len(test_data[1])} test samples\n")

    # Run experiments
    all_results = {}

    if args.method == 'all':
        # Compare all methods
        methods = ['fedavg', 'fedprox', 'ifca', 'agglomerative-inter', 'agglomerative-global']

        for method in methods:
            # Reset clients for each method
            clients = [
                FederatedClient(i, data, args.batch_size, args.dataset)
                for i, data in enumerate(fed_data)
            ]

            results = run_method(method, clients, test_data, args)
            all_results[method] = results

            # Save individual results
            save_results(results, args, method)
            if SAVE_CLIENTS_STATS:
                save_client_stats_csv(clients, args, method)

        # Print comparison summary
        print_comparison_summary(all_results)

        # Create summary table
        summary_table = create_summary_table(all_results)
        print(summary_table)

        # Plot comparison
        plot_comparison(all_results, args)
    else:
        # Run single method
        results = run_method(args.method, clients, test_data, args)
        all_results[args.method] = results

        # Save results
        save_results(results, args, args.method)
        if SAVE_CLIENTS_STATS:
            save_client_stats_csv(clients, args, args.method)

        # Print final results
        print_final_results(results, args.method)


def print_comparison_summary(all_results):
    """Print comparison summary for all methods"""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    for method, results in all_results.items():
        final_acc = results['accuracy_history'][-1]
        avg_time = np.mean(results['round_times'])
        total_time = sum(results['round_times'])

        print(f"\n{method.upper()}:")
        print(f"  Final Accuracy: {final_acc:.4f}")
        print(f"  Avg Time/Round: {avg_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")


def print_final_results(results, method_name):
    """Print final results for a single method"""
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS - {method_name.upper()}")
    print("=" * 60)

    final_acc = results['accuracy_history'][-1]
    avg_time = np.mean(results['round_times'])
    total_time = sum(results['round_times'])

    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Avg Time/Round: {avg_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()