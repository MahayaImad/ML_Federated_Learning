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

from data_preparation import prepare_federated_cifar10, prepare_federated_cifar100, prepare_federated_mnist, \
    number_classes
from models import create_model, initialize_global_model, initialize_edge_models, get_trainable_weights, \
    set_trainable_weights
from hierarchical_server import setup_vanilla_fl, setup_standard_hierarchy, setup_dropin_hierarchy, \
    setup_agglomerative_hierarchy
from client import FederatedClient
from aggregator import FedAvgAggregator
from utils import setup_gpu, save_results, save_client_stats_csv
from config import SAVE_CLIENTS_STATS, VERBOSE, LOCAL_EPOCHS, COMMUNICATION_ROUNDS, BATCH_SIZE, CLIENTS, EDGE_SERVERS


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='FL Comparison: Vanilla vs Hierarchical vs Drop-in vs Agglomerative (with DenseNet Transfer Learning for CIFAR-10)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available training types:

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

  🔹 CIFAR-10 with DenseNet Transfer Learning:
     python main.py --hierarchy-type vanilla --dataset cifar10 --clients 20 --rounds 20
     (Automatically uses DenseNet with frozen base, communicates only trainable layers)

  🔹 Complete comparison:
     python main.py --hierarchy-type compare --dataset mnist \
    --clients 20 --edge-servers 5 --rounds 20 
        """
    )

    # Required arguments
    parser.add_argument('--hierarchy-type', type=str, required=True,
                        choices=['vanilla', 'hierarchical', 'agglomerative', 'drop-in', 'compare'],
                        help='Hierarchical training type')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar10', 'cifar100'],
                        help='Dataset (cifar10 uses DenseNet transfer learning)')

    # Optional arguments
    parser.add_argument('--clients', type=int, default=CLIENTS,
                        help='Total number of clients (default: 20)')

    parser.add_argument('--edge-servers', type=int, default=EDGE_SERVERS,
                        help='Number of edge servers for hierarchy (default: 5)')

    parser.add_argument('--js_threshold', type=float, default=0.5,
                        help='JS threshold for agglomerative clustering')

    parser.add_argument('--client_selection_ratio', type=float, default=0.3,
                        help='Client selection ratio per cluster (e.g., 0.3 = 30%)')

    parser.add_argument('--local-epochs', type=int, default=LOCAL_EPOCHS,
                        help='Local epochs per client (default: 5)')

    parser.add_argument('--rounds', type=int, default=COMMUNICATION_ROUNDS,
                        help='Communication rounds (default: 20)')

    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size (default: 64)')

    parser.add_argument('--iid', action='store_true',
                        help='IID data distribution (default: non-IID)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use (-1 for CPU, default: 0)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    return parser.parse_args()


def setup_hierarchy(clients_data, args):
    """Setup hierarchy based on type"""
    num_classes = number_classes(args.dataset)
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
    """Vanilla FL training (FedAvg) with transfer learning support"""
    print(f" Training Vanilla FL (FedAvg){'with DenseNet Transfer Learning' if args.dataset == 'cifar10' else ''}...")

    global_model = initialize_global_model(args.dataset)
    aggregator = FedAvgAggregator()

    # Check if using transfer learning
    use_transfer = (args.dataset == 'cifar10')

    if use_transfer:
        print(f"  🔹 Using DenseNet121 with transfer learning")
        trainable_weights, trainable_indices = get_trainable_weights(global_model)
        total_params = sum([w.size for w in global_model.get_weights()])
        trainable_params = sum([w.size for w in trainable_weights])
        print(f"  🔹 Total parameters: {total_params:,}")
        print(f"  🔹 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
        print(f"  🔹 Communication reduction: {100 * (1 - trainable_params / total_params):.1f}%")

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

            if use_transfer:
                # Get only trainable weights
                weights = client.get_trainable_update()
            else:
                # Get all weights
                weights = client.local_model.get_weights()

            client_updates.append(weights)
            client_sizes.append(len(client.x_train))

        # Aggregation
        if use_transfer:
            # Aggregate only trainable weights
            aggregated_trainable = aggregator.aggregate_trainable_only(client_updates, client_sizes)
            _, trainable_indices = get_trainable_weights(global_model)
            set_trainable_weights(global_model, aggregated_trainable, trainable_indices)
        else:
            # Standard FedAvg aggregation
            global_weights = aggregator.aggregate(client_updates, client_sizes, global_model)
            global_model.set_weights(global_weights)

        # Evaluation
        x_test, y_test = test_data
        # y_test is already one-hot encoded, no need to convert
        _, accuracy = global_model.evaluate(x_test, y_test, verbose=0)

        # Metrics
        comm_cost = len(clients) * 2  # Upload + download for each client
        if use_transfer:
            # Adjust communication cost for transfer learning
            _, trainable_indices = get_trainable_weights(global_model)
            comm_cost = int(comm_cost * len(trainable_indices) / len(global_model.get_weights()))

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
    """Hierarchical FL training with transfer learning support"""
    print(
        f" Training FL {args.hierarchy_type}{'with DenseNet Transfer Learning' if args.dataset == 'cifar10' else ''}...")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    # Check if using transfer learning
    use_transfer = (args.dataset == 'cifar10')

    if use_transfer:
        print(f"  🔹 Using DenseNet121 with transfer learning")
        trainable_weights, trainable_indices = get_trainable_weights(global_model)
        total_params = sum([w.size for w in global_model.get_weights()])
        trainable_params = sum([w.size for w in trainable_weights])
        print(f"  🔹 Total parameters: {total_params:,}")
        print(f"  🔹 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
        print(f"  🔹 Communication reduction: {100 * (1 - trainable_params / total_params):.1f}%")

    accuracy_history = []
    communication_costs = []
    round_times = []

    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0

        # STEP 1: Broadcast global model to edge servers
        for edge_server in edge_servers:
            if use_transfer:
                # Send only trainable weights
                trainable_weights, trainable_indices = get_trainable_weights(global_model)
                set_trainable_weights(edge_server.local_model, trainable_weights, trainable_indices)
                communication_cost += len(trainable_indices)
            else:
                edge_server.set_global_model(global_model.get_weights())
                communication_cost += 1  # Global -> Edge

        # STEP 2: Each edge server trains clients and aggregates locally
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            # Get clients for this edge server
            edge_clients = [client for client in clients if client.client_id in edge_server.client_ids]

            if not edge_clients:
                continue

            # Local training for clients of this edge server
            client_updates = []
            client_data_sizes = []

            for client in edge_clients:
                # Update with edge server model
                client.update_model(edge_server.local_model)

                # Local training
                client.train_local(args.local_epochs)

                if use_transfer:
                    # Get only trainable weights
                    client_updates.append(client.get_trainable_update())
                    communication_cost += len(get_trainable_weights(client.local_model)[1])
                else:
                    client_updates.append(client.local_model.get_weights())
                    communication_cost += 2  # Edge -> Client -> Edge

                client_data_sizes.append(len(client.x_train))

            # STEP 3: Aggregation at edge server level
            if client_updates:
                if use_transfer:
                    edge_aggregated = fedavg_aggregate(client_updates, client_data_sizes)
                    _, trainable_indices = get_trainable_weights(edge_server.local_model)
                    set_trainable_weights(edge_server.local_model, edge_aggregated, trainable_indices)
                    edge_updates.append(edge_aggregated)
                else:
                    edge_aggregated = fedavg_aggregate(client_updates, client_data_sizes)
                    edge_server.local_model.set_weights(edge_aggregated)
                    edge_updates.append(edge_aggregated)

                edge_weights.append(sum(client_data_sizes))

        # STEP 4: Final aggregation at global server
        if edge_updates:
            global_aggregated = fedavg_aggregate(edge_updates, edge_weights)

            if use_transfer:
                _, trainable_indices = get_trainable_weights(global_model)
                set_trainable_weights(global_model, global_aggregated, trainable_indices)
                communication_cost += len(edge_servers) * len(trainable_indices)
            else:
                global_model.set_weights(global_aggregated)
                communication_cost += len(edge_servers)

        # STEP 5: Evaluation
        x_test, y_test = test_data
        # y_test is already one-hot encoded, no need to convert
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
        'method': args.hierarchy_type,
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'round_times': round_times
    }


def fedavg_aggregate(model_weights_list, data_sizes):
    """FedAvg aggregation weighted by data size"""
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
    """Compare all FL methods"""
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
    """Train a single method"""
    # Create clients
    clients = [FederatedClient(i, data, FedAvgAggregator(), args.batch_size, args.dataset)
               for i, data in enumerate(fed_data)]

    # Train based on type
    if args.hierarchy_type == 'vanilla':
        return train_vanilla_fl(clients, test_data, args)
    else:
        # All hierarchical variants use same training logic
        edge_servers, hierarchical_server = setup_hierarchy(fed_data, args)
        return train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args)


def print_final_results(results, hierarchy_type):
    """Display final results"""
    print("\n" + "=" * 60)
    print(f"=== FINAL RESULTS - {hierarchy_type.upper()} ===")
    print("=" * 60)

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
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()

    # Main startup
    print("Starting federated learning experiment...")
    print(f"Mode: {args.hierarchy_type}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == 'cifar10':
        print("  ⚡ Using DenseNet121 transfer learning (frozen base, trainable top)")
        print("  ⚡ Only trainable weights will be communicated")
    print(f"Configuration: {args.clients} clients, {args.rounds} rounds")
    print(f"Data distribution: {'IID' if args.iid else 'Non-IID'}")

    # Setup GPU
    setup_gpu(args.gpu)

    # Set seed
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

    # Execute based on type
    results = {}
    if args.hierarchy_type == 'compare':
        # Compare all methods
        results = compare_all_methods(fed_data, test_data, args)
    else:
        results = _train_single_method(fed_data, test_data, args)
        results['method'] = args.hierarchy_type

    # Save and visualize
    save_results(results, args)
    print_final_results(results, args.hierarchy_type)


if __name__ == "__main__":
    main()