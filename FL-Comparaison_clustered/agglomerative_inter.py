"""
Agglomerative Clustering FL Training - With Inter-Cluster Aggregation
Variant with cyclic inter-cluster communication
"""

import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from models import initialize_global_model
from communication_cost import CommunicationTracker
from client_selection import select_clients_in_clusters, build_djs_clusters
from fedavg import fedavg_aggregate
from config import VERBOSE


def train_agglomerative_inter(clients, clients_data, test_data, args):
    """
    Train using Agglomerative Clustering with Inter-Cluster Aggregation

    Args:
        clients: List of FederatedClient objects
        clients_data: List of (x_train, y_train) tuples
        test_data: (x_test, y_test) tuple
        args: Command line arguments

    Returns:
        results: Dictionary with training metrics
    """
    print("Training with Agglomerative Clustering (DJS-based + Inter-cluster aggregation)...")

    # Initialize communication tracker
    comm_tracker = CommunicationTracker(args.dataset)

    # Get number of classes
    num_classes = clients_data[0][1].shape[1]

    # Build clusters using Jensen-Shannon divergence
    print("Building clusters using DJS...")
    clusters, cluster_histograms = build_djs_clusters(
        clients_data,
        args.js_threshold,
        num_classes,
        args.verbose
    )

    num_clusters = len(clusters)
    print(f"Created {num_clusters} clusters")
    for cluster_id, client_ids in clusters.items():
        print(f"  Cluster {cluster_id}: {len(client_ids)} clients")

    # Initialize global model
    global_model = initialize_global_model(args.dataset)

    # Initialize cluster models
    cluster_models = {
        cluster_id: initialize_global_model(args.dataset)
        for cluster_id in clusters.keys()
    }

    # Training history
    accuracy_history = []
    loss_history = []
    round_times = []
    communication_costs = []

    # Select clients for training based on selection ratio and mode
    selected_clients_per_cluster = select_clients_in_clusters(
        clusters, clients_data, args.selection_ratio,
        selection_mode=args.client_selection,
        num_classes=num_classes,
        alpha=getattr(args, 'selection_alpha', 0.5)
    )

    print(f"\nClient selection mode: {args.client_selection}")
    print(f"Selection ratio: {args.selection_ratio}")
    if args.client_selection == 'balanced':
        print(f"Selection alpha (size←0.5→diversity): {getattr(args, 'selection_alpha', 0.5)}")
    print(f"Inter-cluster aggregation: ENABLED")

    # Training loop
    for round_num in range(args.rounds):
        round_start = time.time()
        comm_tracker.reset_round()

        if args.verbose or round_num % 5 == 0:
            print(f"\nRound {round_num + 1}/{args.rounds}")

        # Re-select clients if random mode
        if args.client_selection == 'random' and round_num > 0:
            selected_clients_per_cluster = select_clients_in_clusters(
                clusters, clients_data, args.selection_ratio,
                selection_mode='random',
                num_classes=num_classes
            )

        # Count total selected clients for communication tracking
        total_selected_clients = sum(len(clients) for clients in selected_clients_per_cluster.values())

        # STEP 0: Broadcast cluster models to clients
        comm_tracker.record_client_download(total_selected_clients)

        # STEP 1: Train clients in each cluster with their cluster model
        cluster_updates = {}
        cluster_weights = {}

        for cluster_id, client_ids in selected_clients_per_cluster.items():
            # Get cluster model weights
            cluster_model_weights = cluster_models[cluster_id].get_weights()

            client_updates = []
            client_sizes = []

            for client_id in client_ids:
                client = clients[client_id]

                # Update client model with cluster model
                client.local_model.set_weights(cluster_model_weights)

                # Local training
                client.train_local(args.local_epochs)

                # Collect update
                client_updates.append(client.local_model.get_weights())
                client_sizes.append(len(client.x_train))

            # Aggregate within cluster
            if client_updates:
                cluster_aggregated = fedavg_aggregate(client_updates, client_sizes)
                cluster_updates[cluster_id] = cluster_aggregated
                cluster_weights[cluster_id] = sum(client_sizes)

                # Update cluster model
                cluster_models[cluster_id].set_weights(cluster_aggregated)

        # Clients upload to cluster servers
        comm_tracker.record_client_upload(total_selected_clients)

        # STEP 2: Inter-cluster aggregation (cyclic)
        # Each cluster exchanges information with other clusters
        if num_clusters > 1:
            cluster_ids = list(clusters.keys())
            aggregated_cluster_models = {}

            for cluster_id in cluster_ids:
                # Get cyclic partner for this cluster
                partner_id = get_cyclic_partner(cluster_id, round_num, num_clusters)

                if args.verbose:
                    print(f"  Cluster {cluster_id} <-> Cluster {partner_id}")

                # Aggregate current cluster with its partner
                models_to_aggregate = [
                    cluster_models[cluster_id].get_weights(),
                    cluster_models[partner_id].get_weights()
                ]
                weights_to_aggregate = [
                    cluster_weights.get(cluster_id, 1),
                    cluster_weights.get(partner_id, 1)
                ]

                # Inter-cluster aggregation
                inter_cluster_weights = fedavg_aggregate(
                    models_to_aggregate,
                    weights_to_aggregate
                )
                aggregated_cluster_models[cluster_id] = inter_cluster_weights

            # Update all cluster models with inter-cluster aggregation
            for cluster_id, weights in aggregated_cluster_models.items():
                cluster_models[cluster_id].set_weights(weights)

            # Record inter-cluster communications (each cluster communicates with partner)
            comm_tracker.record_inter_cluster(num_clusters)

        # Cluster servers upload to global server
        comm_tracker.record_cluster_upload(num_clusters)

        # STEP 3: Global aggregation across all clusters
        if cluster_updates:
            global_weights = fedavg_aggregate(
                [cluster_models[cid].get_weights() for cid in cluster_models.keys()],
                [cluster_weights.get(cid, 1) for cid in cluster_models.keys()]
            )
            global_model.set_weights(global_weights)

        # Global server broadcasts back to cluster servers
        comm_tracker.record_cluster_download(num_clusters)

        # Evaluation
        x_test, y_test = test_data
        y_test_labels = np.argmax(y_test, axis=1)
        loss, accuracy = global_model.evaluate(x_test, y_test_labels, verbose=0)

        # Finalize communication cost for this round
        round_comm_cost = comm_tracker.finalize_round()

        round_time = time.time() - round_start

        accuracy_history.append(accuracy)
        loss_history.append(loss)
        round_times.append(round_time)
        communication_costs.append(round_comm_cost)

        if args.verbose:
            print(f"  Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, Time: {round_time:.2f}s, Comm: {round_comm_cost:.2f} KB")

    # Print communication summary
    comm_tracker.print_summary()

    return {
        'accuracy_history': accuracy_history,
        'loss_history': loss_history,
        'round_times': round_times,
        'communication_costs': communication_costs,
        'communication_tracker': comm_tracker,
        'num_clusters': len(clusters),
        'clusters': clusters,
        'variant': 'inter-cluster'
    }


def get_cyclic_partner(cluster_id, round_num, num_clusters):
    """
    Calculate the cyclic partner for a cluster at a given round

    Args:
        cluster_id: Current cluster ID
        round_num: Current round number
        num_clusters: Total number of clusters

    Returns:
        partner_id: ID of the partner cluster
    """
    if num_clusters <= 1:
        return cluster_id

    # Cyclic offset increases with each round
    offset = (round_num % (num_clusters - 1)) + 1
    partner_id = (cluster_id + offset) % num_clusters

    return partner_id

