import argparse
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import time

from data_preparation import prepare_federated_cifar10,prepare_federated_cifar100, prepare_federated_mnist, number_classes
from models import create_model, initialize_global_model, initialize_edge_models
from hierarchical_server import setup_vanilla_fl, setup_standard_hierarchy, setup_dropin_hierarchy, setup_agglomerative_hierarchy
from client import FederatedClient
from aggregator import FedAvgAggregator
from utils import setup_gpu, save_results, save_client_stats_csv
from config import SAVE_CLIENTS_STATS, VERBOSE, LOCAL_EPOCHS, COMMUNICATION_ROUNDS, BATCH_SIZE, LEARNING_RATE, CLIENTS, EDGE_SERVERS


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Comparaison FL Vanilla vs Hierarchical vs Drop-in vs agglomerative',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Types d'entraînement disponibles:

  🔹 Vanilla FL:
     python main.py --hierarchy-type vanilla --dataset mnist --clients 20 --rounds 20

  🔹 Hierarchical FL:
     python main.py --hierarchy-type hierarchical --dataset cifar10 \
    --clients 20 --edge-servers 5 --rounds 20

  🔹 Drop-in Hierarchical FL:
     python main.py --hierarchy-type drop-in --clients 20 --edge-servers 5
     
  🔹 Agglomerative clustering (data-driven)
     python main.py --hierarchy-type agglomerative --dataset cifar100 \
     --clients 30 --js-threshold 0.5 --selection-ratio 0.3 --rounds 25

  🔹 Comparaison complète:
     python main.py --hierarchy-type compare --dataset mnist \
    --clients 20 --edge-servers 5 --rounds 20 
        """
    )

    # Arguments obligatoires
    parser.add_argument('--hierarchy-type', type=str, required=True,
                        choices=['vanilla', 'hierarchical', 'agglomerative', 'drop-in', 'compare'],
                        help='Type d\'entraînement hiérarchique')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar10', 'cifar100'],
                        help='Dataset à utiliser (mnist: 28x28 N&B, cifar: 32x32 couleur)')

    # Arguments optionnels
    parser.add_argument('--clients', type=int, default=CLIENTS,
                        help='Nombre total de clients (défaut: 20)')

    parser.add_argument('--edge-servers', type=int, default=EDGE_SERVERS,
                        help='Nombre d\'edge servers pour hiérarchie (défaut: 5)')

    parser.add_argument('--js_threshold', type=float, default=0.5,
                        help='Seuil pour clustering agglomératif JS')

    parser.add_argument('--client_selection_ratio', type=float, default=0.3,
                        help='Pourcentage de clients à sélectionner par cluster (ex: 0.3 = 30%)')

    parser.add_argument('--local-epochs', type=int, default=LOCAL_EPOCHS,
                        help='Époques locales par client (défaut: 5)')

    parser.add_argument('--rounds', type=int, default=COMMUNICATION_ROUNDS,
                        help='Nombre de rounds de communication (défaut: 20)')

    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Taille des lots (défaut: 32)')

    parser.add_argument('--iid', action='store_true',
                        help='Distribution IID des données (défaut: non-IID)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU à utiliser (-1 pour CPU, défaut: 0)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire (défaut: 42)')

    return parser.parse_args()


def setup_hierarchy(clients_data, args):

    num_classes = number_classes()
    hierarchy_type = args.hierarchy_type

    if hierarchy_type == 'vanilla':
        return setup_vanilla_fl()

    elif hierarchy_type == 'hierarchical':
        return setup_standard_hierarchy(
            clients_data, args.edge_servers, VERBOSE
        )

    elif hierarchy_type == 'agglomerative':
        return setup_agglomerative_hierarchy(
            clients_data,
            args.js_threshold,
            args.client_selection_ratio,
            num_classes,
            VERBOSE
        )

    elif hierarchy_type == 'drop-in':
        return setup_dropin_hierarchy(
            clients_data, args.edge_servers, VERBOSE
        )

    else:
        print(f"Error: Unknown hierarchy type '{hierarchy_type}'")
        return None, None


def train_vanilla_fl(clients, test_data, args):
    """Entraînement FL vanilla (FedAvg classique)"""
    print(" Train of Vanilla FL (FedAvg)...")

    global_model =  initialize_global_model(args.dataset)
    aggregator = FedAvgAggregator()

    results = {
        'accuracy_history': [],
        'communication_costs': [],
        'round_times': []
    }

    for round_num in range(args.rounds):
        start_time = time.time()

        if VERBOSE:
            print(f"  Round {round_num + 1}/{args.rounds}")

        # Local training
        client_updates = []
        client_sizes = []

        for client in clients:
            client.update_model(global_model)
            client.train_local(args.local_epochs)
            weights = client.local_model.get_weights()
            client_updates.append(weights)
            client_sizes.append(len(client.x_train))

        # Aggregation FedAvg
        global_weights = aggregator.aggregate(client_updates, client_sizes, global_model)
        global_model.set_weights(global_weights)

        # Evaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        _, accuracy = global_model.evaluate(x_test, y_test, verbose=0)

        # Métriques
        comm_cost = len(clients) * 2  # Upload + download pour chaque client
        round_time = time.time() - start_time

        results['accuracy_history'].append(accuracy)
        results['communication_costs'].append(comm_cost)
        results['round_times'].append(round_time)

        if VERBOSE:
            print(f"    Accuracy: {accuracy:.4f}, Comm Cost: {comm_cost}, Time: {round_time:.2f}s")

    # Save client statistics to CSV
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return results


def train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args):

    print(f" Train of FL {args.hierarchy_type}...")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    accuracy_history = []
    communication_costs = []
    round_times = []

    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0

        # ÉTAPE 1: Diffuser le modèle global vers les edge servers
        for edge_server in edge_servers:
            edge_server.set_global_model(global_model.get_weights())
            communication_cost += 1  # Global -> Edge

        # ÉTAPE 2: Chaque edge server entraîne ses clients et agrège localement
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            # Clients de cet edge server
            edge_clients = [client for client in clients if client.client_id in edge_server.client_ids]

            if not edge_clients:
                continue

            # Entraînement local des clients de cet edge server
            client_updates = []
            client_data_sizes = []

            for client in edge_clients:
                # Mettre à jour avec le modèle de l'edge server
                client.update_model(edge_server.local_model)

                # Entraînement local
                client.train_local(args.local_epochs)
                communication_cost += 2  # Edge -> Client -> Edge

                client_updates.append(client.local_model.get_weights())
                client_data_sizes.append(len(client.x_train))

            # ÉTAPE 3: Agrégation au niveau de l'edge server
            if client_updates:
                edge_aggregated = fedavg_aggregate(client_updates, client_data_sizes)
                edge_server.local_model.set_weights(edge_aggregated)

                edge_updates.append(edge_aggregated)
                edge_weights.append(sum(client_data_sizes))  # Poids = total des données

        # ÉTAPE 4: Agrégation finale au serveur global
        if edge_updates:
            global_aggregated = fedavg_aggregate(edge_updates, edge_weights)
            global_model.set_weights(global_aggregated)

            # Communication Edge -> Global
            communication_cost += len(edge_servers)

        # ÉTAPE 5: Évaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)

        round_time = time.time() - round_start_time

        accuracy_history.append(test_acc)
        communication_costs.append(communication_cost)
        round_times.append(round_time)

        print(f"  Accuracy: {test_acc:.4f}, Time: {round_time:.2f}s, Comm: {communication_cost}")

    # Save client statistics to CSV
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return {
        'method': 'hierarchical',
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'round_times': round_times
    }


def fedavg_aggregate(model_weights_list, data_sizes):
    """Agrégation FedAvg pondérée par la taille des données"""
    if not model_weights_list:
        return None

    total_size = sum(data_sizes)
    weights_avg = []

    # Weight each model by its data size
    for layer_idx in range(len(model_weights_list[0])):
        weighted_layer = sum(
            (data_sizes[i] / total_size) * model_weights_list[i][layer_idx]
            for i in range(len(model_weights_list))
        )
        weights_avg.append(weighted_layer)

    return weights_avg


def compare_all_methods(fed_data, test_data, args):

    print("\n" + "=" * 60)
    print("Comparing all FL methods...")
    print("=" * 60)

    all_results = {}

    # Define methods to compare
    methods = [
        ('vanilla', 'vanilla'),
        ('hierarchical', 'hierarchical'),
        ('drop_in', 'drop-in'),
        ('agglomerative', 'agglomerative')
    ]

    for method_name, hierarchy_type in methods:
        print(f"\n{'=' * 60}")
        print(f"Training: {method_name.upper()}")
        print(f"{'=' * 60}")

        # Update args for this method
        args.hierarchy_type = hierarchy_type

        # Run training
        results = _train_single_method(fed_data, test_data, args)
        all_results[method_name] = results

        # Save individual results
        save_results({method_name: results}, args)

    return all_results


def _train_single_method(fed_data, test_data, args):

    # Create clients
    clients = [FederatedClient(i, data, FedAvgAggregator(), args.dataset_name)
               for i, data in enumerate(fed_data)]

    # Train based on type
    if args.hierarchy_type == 'vanilla':
        return train_vanilla_fl(clients, test_data, args)
    else:
        # All hierarchical variants use same training logic
        edge_servers, hierarchical_server = setup_hierarchy(fed_data, args)
        return train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args)


def print_final_results(results, hierarchy_type):
    """Affiche les résultats finaux"""
    print("\n" + "="*60)
    print(f"=== FINAL RESULTS - {hierarchy_type.upper()} ===")
    print("="*60)

    if hierarchy_type == 'compare':
        for method, method_results in results.items():
            final_acc = method_results['accuracy_history'][-1]
            avg_time = np.mean(method_results['round_times'])
            total_comm = sum(method_results.get('total_comm_costs', method_results.get('communication_costs', [])))

            print(f"{method.capitalize()}:")
            print(f"  - Final Accuracy: {final_acc:.4f}")
            print(f"  - Time per/round: {avg_time:.2f}s")
            print(f"  - Total Communication Cost: {total_comm}")
            print()
    else:
        final_acc = results['accuracy_history'][-1]
        avg_time = np.mean(results['round_times'])

        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Time per/round: {avg_time:.2f}s")

        if 'total_comm_costs' in results:
            total_comm = sum(results['total_comm_costs'])
            avg_edge_comm = np.mean(results['edge_comm_costs'])
            avg_global_comm = np.mean(results['global_comm_costs'])

            print(f"Total Communication Cost: {total_comm}")
            print(f"Edge Communication Cost mean/round: {avg_edge_comm:.1f}")
            print(f"Global Communication Cost mean/round: {avg_global_comm:.1f}")
        else:
            total_comm = sum(results['communication_costs'])
            print(f"Total Communication Cost: {total_comm}")


def main():

    # Parse des arguments
    args = parse_arguments()

    # Main startup
    print("Starting federated learning experiment...")
    print(f"Mode: {args.hierarchy_type}")
    print(f"Configuration: {args.clients} clients, {args.rounds} rounds")
    print(f"Data distribution: {'IID' if args.iid else 'Non-IID'}")

    # Setup GPU
    setup_gpu(args.gpu)

    # define Seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Data preparation
    print("\nPreparing data...")
    fed_data, test_data = None, None
    if args.dataset == 'cifar10':
        fed_data, test_data, _ = prepare_federated_cifar10(iid=args.iid, num_clients=args.clients)
        print(f"Setup complete: {len(fed_data)} clients, {len(test_data[0])} test samples")

    elif args.dataset == 'cifar100':
        fed_data, test_data, _ = prepare_federated_cifar100(iid=args.iid, num_clients=args.clients)
        print(f"Setup complete: {len(fed_data)} clients, {len(test_data[0])} test samples")

    elif args.dataset == 'mnist':
        fed_data, test_data, _ = prepare_federated_mnist(iid=args.iid, num_clients=args.clients)
        print(f"Setup complete: {len(fed_data)} clients, {len(test_data[0])} test samples")

    # Exécution selon le type
    results = {}
    if args.hierarchy_type == 'compare':
        # Comparaison de toutes les méthodes
        results = compare_all_methods(fed_data, test_data, args)
    else:
        results = _train_single_method(fed_data, test_data, args)
        results['method'] = args.hierarchy_type

    # Sauvegarde et visualisation
    save_results(results, args)
    print_final_results(results, args.hierarchy_type)


if __name__ == "__main__":
    main()