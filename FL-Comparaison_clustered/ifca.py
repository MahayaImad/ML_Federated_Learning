"""
IFCA (Iterative Federated Clustering Algorithm) Training
Learns multiple cluster models and assigns clients based on loss
"""

import time
import numpy as np
from models import initialize_global_model
from communication_cost import CommunicationTracker
from config import VERBOSE


def train_ifca(clients, test_data, args):
    """
    Train using IFCA algorithm

    Args:
        clients: List of FederatedClient objects
        test_data: (x_test, y_test) tuple
        args: Command line arguments

    Returns:
        results: Dictionary with training metrics
    """
    print(f"Training with IFCA (K={args.num_clusters} clusters)...")

    # Initialize communication tracker
    comm_tracker = CommunicationTracker(args.dataset)

    num_clusters = args.num_clusters
    num_clients = len(clients)

    # Initialize K cluster models
    cluster_models = [
        initialize_global_model(args.dataset)
        for _ in range(num_clusters)
    ]

    # Initialize cluster assignments randomly
    client_clusters = np.random.randint(0, num_clusters, len(clients))

    # Training history
    accuracy_history = []
    loss_history = []
    round_times = []
    cluster_history = []
    communication_costs = []

    # Training loop
    for round_num in range(args.rounds):
        round_start = time.time()
        comm_tracker.reset_round()

        if args.verbose or round_num % 5 == 0:
            print(f"\nRound {round_num + 1}/{args.rounds}")

        # Server broadcasts K models to all clients (K * N communications)
        comm_tracker.record_client_download(num_clients * num_clusters)

        # Step 1: Client updates with assigned cluster model
        cluster_updates = {k: [] for k in range(num_clusters)}
        cluster_sizes = {k: [] for k in range(num_clusters)}

        for client_idx, client in enumerate(clients):
            assigned_cluster = client_clusters[client_idx]

            # Update client with assigned cluster model
            client.local_model.set_weights(
                cluster_models[assigned_cluster].get_weights()
            )

            # Local training
            client.train_local(args.local_epochs)

            # Collect update
            cluster_updates[assigned_cluster].append(
                client.local_model.get_weights()
            )
            cluster_sizes[assigned_cluster].append(len(client.x_train))

        # Clients upload to server (each client uploads once)
        comm_tracker.record_client_upload(num_clients)

        # Step 2: Aggregate within each cluster
        for k in range(num_clusters):
            if cluster_updates[k]:
                aggregated = fedavg_aggregate(
                    cluster_updates[k],
                    cluster_sizes[k]
                )
                cluster_models[k].set_weights(aggregated)

        # Step 3: Reassign clients based on loss
        for client_idx, client in enumerate(clients):
            best_cluster = find_best_cluster(
                client,
                cluster_models
            )
            client_clusters[client_idx] = best_cluster

        # Evaluation using best performing cluster model
        best_model = select_best_model(cluster_models, test_data)
        x_test, y_test = test_data
        y_test_labels = np.argmax(y_test, axis=1)
        loss, accuracy = best_model.evaluate(x_test, y_test_labels, verbose=0)

        # Finalize communication cost for this round
        round_comm_cost = comm_tracker.finalize_round()

        round_time = time.time() - round_start

        accuracy_history.append(accuracy)
        loss_history.append(loss)
        round_times.append(round_time)
        cluster_history.append(client_clusters.copy())
        communication_costs.append(round_comm_cost)

        if args.verbose:
            cluster_counts = np.bincount(client_clusters, minlength=num_clusters)
            print(f"  Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, Time: {round_time:.2f}s, Comm: {round_comm_cost:.2f} KB")
            print(f"  Cluster distribution: {cluster_counts}")

    # Print communication summary
    comm_tracker.print_summary()

    return {
        'accuracy_history': accuracy_history,
        'loss_history': loss_history,
        'round_times': round_times,
        'cluster_history': cluster_history,
        'communication_costs': communication_costs,
        'communication_tracker': comm_tracker,
        'num_clusters': num_clusters
    }


def find_best_cluster(client, cluster_models):
    """
    Find the best cluster for a client based on validation loss

    Args:
        client: FederatedClient object
        cluster_models: List of cluster models

    Returns:
        best_cluster: Index of best cluster
    """
    min_loss = float('inf')
    best_cluster = 0

    # Use a small validation set from client's data
    val_size = min(100, len(client.x_train) // 10)
    if val_size == 0:
        val_size = len(client.x_train)

    x_val = client.x_train[:val_size]
    y_val = client.y_train[:val_size]
    y_val_labels = np.argmax(y_val, axis=1)

    for k, model in enumerate(cluster_models):
        loss, _ = model.evaluate(x_val, y_val_labels, verbose=0)
        if loss < min_loss:
            min_loss = loss
            best_cluster = k

    return best_cluster


def select_best_model(cluster_models, test_data):
    """
    Select the best performing cluster model on test data

    Args:
        cluster_models: List of cluster models
        test_data: (x_test, y_test) tuple

    Returns:
        best_model: Best performing model
    """
    x_test, y_test = test_data
    y_test_labels = np.argmax(y_test, axis=1)

    best_acc = 0
    best_model = cluster_models[0]

    for model in cluster_models:
        _, accuracy = model.evaluate(x_test, y_test_labels, verbose=0)
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model

    return best_model


def fedavg_aggregate(model_weights_list, data_sizes):
    """FedAvg aggregation weighted by data sizes"""
    if not model_weights_list:
        return None

    total_size = sum(data_sizes)
    weights_avg = []

    for layer_idx in range(len(model_weights_list[0])):
        weighted_layer = sum(
            (data_sizes[i] / total_size) * model_weights_list[i][layer_idx]
            for i in range(len(model_weights_list))
        )
        weights_avg.append(weighted_layer)

    return weights_avg