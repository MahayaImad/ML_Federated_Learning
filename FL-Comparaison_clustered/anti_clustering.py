"""
AntiClustering FL Training - Maximum Diversity Intra-Cluster
Groups DISSIMILAR clients together for better generalization
"""

import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from models import initialize_global_model
from communication_cost import CommunicationTracker
from client_selection import select_clients_in_clusters, jensen_shannon_distance
from fedavg import fedavg_aggregate
from config import VERBOSE
from config import get_dataset_config


def train_anticlustering(clients, clients_data, test_data, args):
    """
    Train using AntiClustering algorithm - Maximum Diversity Intra-Cluster

    Args:
        clients: List of FederatedClient objects
        clients_data: List of (x_train, y_train) tuples
        test_data: (x_test, y_test) tuple
        args: Command line arguments

    Returns:
        results: Dictionary with training metrics
    """
    print("Training with AntiClustering (Maximum Diversity Intra-Cluster)...")

    # Initialize communication tracker
    comm_tracker = CommunicationTracker(args.dataset)

    # Get number of classes
    num_classes = clients_data[0][1].shape[1]

    # Build anti-clusters using INVERTED Jensen-Shannon divergence
    print("Building anti-clusters using INVERTED DJS (maximizing intra-diversity)...")
    clusters, cluster_histograms, diversity_metrics = build_anticlusters(
        clients_data,
        args.js_threshold,
        num_classes,
        args.verbose
    )

    # Filter out clusters with only 1 client (no intra-cluster diversity possible)
    original_clusters = clusters.copy()
    clusters = {cluster_id: client_ids for cluster_id, client_ids in clusters.items()
                if len(client_ids) > 2}

    ignored_clusters = len(original_clusters) - len(clusters)
    ignored_clients = sum(len(client_ids) for cluster_id, client_ids in original_clusters.items()
                         if len(client_ids) <= 2)

    num_clusters = len(clusters)
    print(f"Created {len(original_clusters)} anti-clusters, using {num_clusters} (ignored {ignored_clusters} single-client clusters)")
    if ignored_clusters > 0:
        print(f"  Ignored {ignored_clients} clients in single-client clusters")

    for cluster_id, client_ids in clusters.items():
        print(f"  Anti-Cluster {cluster_id}: {len(client_ids)} clients (max diversity)")

    # Print diversity metrics
    print_diversity_metrics(diversity_metrics)

    # Initialize global model
    global_model = initialize_global_model(args.dataset)

    # Initialize cluster models only for valid clusters (>1 client)
    cluster_models = {
        cluster_id: initialize_global_model(args.dataset)
        for cluster_id in clusters.keys()  # Only for filtered clusters
    }

    # Check if we have valid clusters for training
    if len(clusters) == 0:
        raise ValueError("No valid anti-clusters found (all clusters have only 1 client). "
                        "Try adjusting --js-threshold or increasing --clients.")

    # Training history
    accuracy_history = []
    loss_history = []
    round_times = []
    communication_costs = []
    diversity_history = []
    cluster_performance_history = []  # Track individual cluster performances

    # Select ALL clients for training (no sub-sampling for pure anti-clustering test)
    # Only for clusters with more than 1 client
    selected_clients_per_cluster = {
        cluster_id: client_ids.copy()  # Select ALL clients in each valid cluster
        for cluster_id, client_ids in clusters.items()
    }

    total_participating_clients = sum(len(clients) for clients in selected_clients_per_cluster.values())
    print(f"\nClient selection: ALL CLIENTS in each anti-cluster (>1 client)")
    print(f"Total clients participating: {total_participating_clients}/{len(clients_data)}")
    print(f"Anti-clustering strategy: MAXIMUM intra-cluster diversity")

    # Training loop
    for round_num in range(args.rounds):
        round_start = time.time()
        comm_tracker.reset_round()

        if args.verbose or round_num % 5 == 0:
            print(f"\nRound {round_num + 1}/{args.rounds}")

        # No re-selection - keep ALL clients throughout training
        # (Commented out for full client participation test)
        # if args.client_selection == 'random' or round_num % 5 == 0:
        #     selected_clients_per_cluster = select_clients_in_clusters(...)

        if args.verbose and round_num == 0:
            print("  Using ALL clients in each anti-cluster (no sub-sampling)")

        # Count total selected clients for communication tracking
        total_selected_clients = sum(len(clients) for clients in selected_clients_per_cluster.values())

        # STEP 1: Broadcast cluster models to clients
        comm_tracker.record_client_download(total_selected_clients)

        # STEP 2: Intra-cluster training with diverse clients
        cluster_updates = {}
        cluster_weights = {}

        for cluster_id, client_ids in selected_clients_per_cluster.items():
            # Get cluster model weights
            cluster_model_weights = cluster_models[cluster_id].get_weights()

            client_updates = []
            client_sizes = []

            if args.verbose:
                print(f"  Anti-Cluster {cluster_id}: Training {len(client_ids)} diverse clients")

            for client_id in client_ids:
                client = clients[client_id]

                # Update client model with cluster model
                client.local_model.set_weights(cluster_model_weights)

                # Local training
                client.train_local(args.local_epochs)

                # Collect update
                client_updates.append(client.local_model.get_weights())
                client_sizes.append(len(client.x_train))

            # Aggregate within anti-cluster (diverse models)
            if client_updates:
                cluster_aggregated = fedavg_aggregate(client_updates, client_sizes)
                cluster_updates[cluster_id] = cluster_aggregated
                cluster_weights[cluster_id] = sum(client_sizes)

                # Update cluster model
                cluster_models[cluster_id].set_weights(cluster_aggregated)

        # STEP 3: Clients upload to cluster servers
        comm_tracker.record_client_upload(total_selected_clients)

        # STEP 4: Cluster servers upload to global server
        comm_tracker.record_cluster_upload(num_clusters)

        # STEP 5: EVALUATION - Test each cluster model individually before aggregation
        if args.verbose or round_num % 5 == 0 or round_num == args.rounds - 1:
            cluster_performances = evaluate_individual_clusters(
                cluster_models, test_data, round_num + 1, args.verbose
            )
        else:
            cluster_performances = {}

        # STEP 6: Central global aggregation
        if cluster_updates:
            global_weights = fedavg_aggregate(
                [cluster_models[cid].get_weights() for cid in cluster_models.keys()],
                [cluster_weights.get(cid, 1) for cid in cluster_models.keys()]
            )
            global_model.set_weights(global_weights)

        # STEP 7: Evaluation of global model after aggregation
        x_test, y_test = test_data
        y_test_labels = np.argmax(y_test, axis=1)
        global_loss, global_accuracy = global_model.evaluate(x_test, y_test_labels, verbose=0)

        # STEP 8: Global server broadcasts back to cluster servers
        comm_tracker.record_cluster_download(num_clusters)

        # Update cluster models with global model for next round
        for cluster_id in cluster_models.keys():
            cluster_models[cluster_id].set_weights(global_model.get_weights())

        # Calculate current diversity metrics
        current_diversity = calculate_round_diversity(
            selected_clients_per_cluster, clients_data, num_classes
        )
        diversity_history.append(current_diversity)

        # Evaluation
        x_test, y_test = test_data
        y_test_labels = np.argmax(y_test, axis=1)
        loss, accuracy = global_model.evaluate(x_test, y_test_labels, verbose=0)

        # Store cluster performances for this round
        cluster_performance_history.append(cluster_performances)

        # Finalize communication cost for this round
        round_comm_cost = comm_tracker.finalize_round()

        round_time = time.time() - round_start

        accuracy_history.append(accuracy)
        loss_history.append(loss)
        round_times.append(round_time)
        communication_costs.append(round_comm_cost)

        # Print detailed results
        if args.verbose or round_num % 5 == 0:
            print_round_analysis(
                round_num + 1, accuracy, loss, round_time, round_comm_cost,
                current_diversity, cluster_performances
            )

    # Print communication summary
    comm_tracker.print_summary()

    # Print final diversity and performance analysis
    print_final_diversity_analysis(diversity_history, clusters, clients_data, num_classes, ignored_clients)
    print_final_cluster_analysis(cluster_performance_history, clusters, clients_data, args)

    return {
        'accuracy_history': accuracy_history,
        'loss_history': loss_history,
        'round_times': round_times,
        'communication_costs': communication_costs,
        'communication_tracker': comm_tracker,
        'num_clusters': len(clusters),
        'clusters': clusters,
        'diversity_metrics': diversity_metrics,
        'diversity_history': diversity_history,
        'cluster_performance_history': cluster_performance_history,
        'variant': 'anticlustering',
        'ignored_clients': ignored_clients,
        'participating_clients': sum(len(client_ids) for client_ids in clusters.values()),
        'total_clients': len(clients_data)
    }


def build_anticlusters(clients_data, js_threshold, num_classes, verbose=False):
    """
    Build anti-clusters using INVERTED Jensen-Shannon divergence
    Goal: Group DISSIMILAR clients together for maximum diversity

    Args:
        clients_data: List of (x_train, y_train) tuples
        js_threshold: Distance threshold for clustering (on inverted matrix)
        num_classes: Number of classes
        verbose: Print verbose output

    Returns:
        clusters: Dict mapping cluster_id to list of client_ids
        histograms: List of normalized histograms for each client
        diversity_metrics: Dictionary with diversity analysis
    """
    # Step 1: Calculate label histograms for each client
    client_histograms = []
    for x_train, y_train in clients_data:
        labels = np.argmax(y_train, axis=1)
        histogram = np.zeros(num_classes)
        unique, counts = np.unique(labels, return_counts=True)
        histogram[unique] = counts
        histogram = histogram / histogram.sum()  # Normalize
        client_histograms.append(histogram)

    # Step 2: Build Jensen-Shannon distance matrix (SIMILARITY)
    n_clients = len(client_histograms)
    js_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            js_dist = jensen_shannon_distance(
                client_histograms[i],
                client_histograms[j]
            )
            js_matrix[i, j] = js_matrix[j, i] = js_dist

    if verbose:
        print_js_statistics(js_matrix, "Original JS Matrix (Similarity)")

    # Step 3: INVERT matrix to get DISSIMILARITY
    max_distance = js_matrix.max()
    js_inverted = max_distance - js_matrix

    # Ensure diagonal is 0 (distance from client to itself)
    np.fill_diagonal(js_inverted, 0)

    if verbose:
        print_js_statistics(js_inverted, "Inverted JS Matrix (Dissimilarity)")

    # Step 4: Agglomerative clustering on INVERTED matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=js_threshold
    )
    cluster_labels = clustering.fit_predict(js_inverted)

    # Step 5: Create clusters dictionary
    clusters = {}
    for client_idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(client_idx)

    # Step 6: Calculate diversity metrics
    diversity_metrics = calculate_diversity_metrics(
        clusters, client_histograms, js_matrix, js_inverted
    )

    return clusters, client_histograms, diversity_metrics


def calculate_diversity_metrics(clusters, client_histograms, js_original, js_inverted):
    """
    Calculate comprehensive diversity metrics for anti-clusters

    Returns:
        Dictionary with diversity analysis
    """
    metrics = {
        'num_clusters': len(clusters),
        'cluster_sizes': [len(clients) for clients in clusters.values()],
        'intra_cluster_diversity': [],
        'inter_cluster_similarity': [],
        'diversity_improvement': 0
    }

    # Calculate intra-cluster diversity (using original distances)
    for cluster_id, client_ids in clusters.items():
        if len(client_ids) > 1:
            cluster_distances = []
            for i in range(len(client_ids)):
                for j in range(i + 1, len(client_ids)):
                    dist = js_original[client_ids[i], client_ids[j]]
                    cluster_distances.append(dist)

            avg_intra_diversity = np.mean(cluster_distances)
            metrics['intra_cluster_diversity'].append(avg_intra_diversity)
        else:
            metrics['intra_cluster_diversity'].append(0)

    # Calculate inter-cluster similarity
    cluster_ids = list(clusters.keys())
    if len(cluster_ids) > 1:
        inter_distances = []
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                # Average distance between clusters
                cluster_i = clusters[cluster_ids[i]]
                cluster_j = clusters[cluster_ids[j]]

                distances = []
                for ci in cluster_i:
                    for cj in cluster_j:
                        distances.append(js_original[ci, cj])

                inter_distances.append(np.mean(distances))

        metrics['inter_cluster_similarity'] = inter_distances

    # Calculate diversity improvement
    avg_intra_original = np.mean(js_original[np.triu_indices_from(js_original, k=1)])
    avg_intra_anticlusters = np.mean(metrics['intra_cluster_diversity'])
    metrics['diversity_improvement'] = (avg_intra_anticlusters - avg_intra_original) / avg_intra_original * 100

    return metrics


def calculate_round_diversity(selected_clients_per_cluster, clients_data, num_classes):
    """
    Calculate diversity for the current round selection

    Returns:
        Average diversity score for this round
    """
    all_selected = []
    for client_ids in selected_clients_per_cluster.values():
        all_selected.extend(client_ids)

    if len(all_selected) < 2:
        return 0

    # Calculate histograms for selected clients
    histograms = []
    for client_id in all_selected:
        _, y_train = clients_data[client_id]
        labels = np.argmax(y_train, axis=1)
        histogram = np.zeros(num_classes)
        unique, counts = np.unique(labels, return_counts=True)
        histogram[unique] = counts
        histogram = histogram / histogram.sum()
        histograms.append(histogram)

    # Calculate average pairwise JS distance
    distances = []
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            dist = jensen_shannon_distance(histograms[i], histograms[j])
            distances.append(dist)

    return np.mean(distances) if distances else 0


def print_js_statistics(js_matrix, title):
    """Print Jensen-Shannon distance statistics"""
    js_values = js_matrix[np.triu_indices_from(js_matrix, k=1)]
    print(f"\n  {title}:")
    print(f"    Min: {js_values.min():.4f}, Max: {js_values.max():.4f}")
    print(f"    Mean: {js_values.mean():.4f}, Median: {np.median(js_values):.4f}")


def print_diversity_metrics(diversity_metrics):
    """Print comprehensive diversity metrics"""
    print(f"\n  === ANTI-CLUSTERING DIVERSITY ANALYSIS ===")
    print(f"  Number of anti-clusters: {diversity_metrics['num_clusters']}")
    print(f"  Cluster sizes: {diversity_metrics['cluster_sizes']}")
    print(f"  Average intra-cluster diversity: {np.mean(diversity_metrics['intra_cluster_diversity']):.4f}")
    if diversity_metrics['inter_cluster_similarity']:
        print(f"  Average inter-cluster similarity: {np.mean(diversity_metrics['inter_cluster_similarity']):.4f}")
    print(f"  Diversity improvement vs random: {diversity_metrics['diversity_improvement']:.2f}%")
    print(f"  ============================================")


def print_final_diversity_analysis(diversity_history, clusters, clients_data, num_classes, ignored_clients=0):
    """Print final analysis of diversity throughout training"""
    print("\n" + "=" * 60)
    print("ANTI-CLUSTERING DIVERSITY ANALYSIS")
    print("=" * 60)


def evaluate_individual_clusters(cluster_models, test_data, round_num, verbose=False):
    """
    Evaluate each cluster model individually on test data

    Args:
        cluster_models: Dict mapping cluster_id to model
        test_data: (x_test, y_test) tuple
        round_num: Current round number
        verbose: Print detailed results

    Returns:
        Dict with cluster performances
    """
    x_test, y_test = test_data
    y_test_labels = np.argmax(y_test, axis=1)

    cluster_performances = {}

    if verbose:
        print(f"\n  === INDIVIDUAL CLUSTER EVALUATIONS (Round {round_num}) ===")

    for cluster_id, model in cluster_models.items():
        loss, accuracy = model.evaluate(x_test, y_test_labels, verbose=0)
        cluster_performances[cluster_id] = {
            'accuracy': accuracy,
            'loss': loss
        }

        if verbose:
            print(f"    Anti-Cluster {cluster_id}: Acc={accuracy:.4f}, Loss={loss:.4f}")

    if verbose:
        # Find best and worst performing clusters
        best_cluster = max(cluster_performances.items(), key=lambda x: x[1]['accuracy'])
        worst_cluster = min(cluster_performances.items(), key=lambda x: x[1]['accuracy'])

        print(f"    Best cluster: {best_cluster[0]} (Acc={best_cluster[1]['accuracy']:.4f})")
        print(f"    Worst cluster: {worst_cluster[0]} (Acc={worst_cluster[1]['accuracy']:.4f})")
        print(f"    Performance gap: {best_cluster[1]['accuracy'] - worst_cluster[1]['accuracy']:.4f}")
        print(f"  ======================================================")

    return cluster_performances


def print_round_analysis(round_num, global_acc, global_loss, round_time, comm_cost,
                        diversity, cluster_performances):
    """
    Print comprehensive round analysis
    """
    print(f"\n  === ROUND {round_num} COMPREHENSIVE ANALYSIS ===")

    # Global model performance
    print(f"  Global Model Performance:")
    print(f"    Accuracy: {global_acc:.4f}, Loss: {global_loss:.4f}")

    # Individual cluster performance analysis
    if cluster_performances:
        accuracies = [perf['accuracy'] for perf in cluster_performances.values()]
        losses = [perf['loss'] for perf in cluster_performances.values()]

        print(f"  Cluster Performance Summary:")
        print(f"    Avg Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
        print(f"    Avg Loss: {np.mean(losses):.4f} (±{np.std(losses):.4f})")
        print(f"    Best Accuracy: {np.max(accuracies):.4f}")
        print(f"    Worst Accuracy: {np.min(accuracies):.4f}")

        # Aggregation effect analysis
        avg_cluster_acc = np.mean(accuracies)
        aggregation_effect = global_acc - avg_cluster_acc
        print(f"  Aggregation Effect:")
        print(f"    Global vs Avg Cluster: {aggregation_effect:+.4f}")
        print(f"    Aggregation {'IMPROVES' if aggregation_effect > 0 else 'DEGRADES'} performance")

    # Other metrics
    print(f"  Other Metrics:")
    print(f"    Diversity: {diversity:.4f}")
    print(f"    Time: {round_time:.2f}s, Comm: {comm_cost:.2f} KB")
    print(f"  ===============================================")


def analyze_final_cluster_performances(cluster_performance_history, clusters):
    """
    Analyze cluster performances across all rounds

    Args:
        cluster_performance_history: List of cluster performances per round
        clusters: Dict mapping cluster_id to client_ids

    Returns:
        Analysis dictionary
    """
    if not cluster_performance_history:
        return {}

    analysis = {}

    # Get all cluster IDs
    cluster_ids = list(clusters.keys())

    for cluster_id in cluster_ids:
        cluster_accs = []
        cluster_losses = []

        for round_perfs in cluster_performance_history:
            if cluster_id in round_perfs:
                cluster_accs.append(round_perfs[cluster_id]['accuracy'])
                cluster_losses.append(round_perfs[cluster_id]['loss'])

        if cluster_accs:  # If we have data for this cluster
            analysis[cluster_id] = {
                'avg_accuracy': np.mean(cluster_accs),
                'std_accuracy': np.std(cluster_accs),
                'final_accuracy': cluster_accs[-1] if cluster_accs else 0,
                'avg_loss': np.mean(cluster_losses),
                'improvement': cluster_accs[-1] - cluster_accs[0] if len(cluster_accs) > 1 else 0,
                'num_clients': len(clusters[cluster_id])
            }

    return analysis


def print_final_cluster_analysis(cluster_performance_history, clusters, clients_data, args):
    """
    Print final analysis of individual cluster performances
    """
    print("\n" + "=" * 60)
    print("INDIVIDUAL CLUSTER PERFORMANCE ANALYSIS")
    print("=" * 60)

    if not cluster_performance_history:
        print("No cluster performance data available.")
        print("=" * 60)
        return

    analysis = analyze_final_cluster_performances(cluster_performance_history, clusters)
    config = get_dataset_config(args.dataset)

    if not analysis:
        print("No cluster analysis data available.")
        print("=" * 60)
        return

    # Sort clusters by final accuracy
    sorted_clusters = sorted(analysis.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)

    print(f"{'Cluster':<8} {'Clients':<8} {'Final Acc':<10} {'Avg Acc':<10} {'Improvement':<12} {'Std Dev':<10}")
    print("-" * 70)

    for cluster_id, stats in sorted_clusters:
        print(f"{cluster_id:<8} {stats['num_clients']:<8} {stats['final_accuracy']:<10.4f} "
              f"{stats['avg_accuracy']:<10.4f} {stats['improvement']:<12.4f} "
              f"{stats['std_accuracy']:<10.4f}")

    # Summary statistics
    final_accs = [stats['final_accuracy'] for stats in analysis.values()]
    improvements = [stats['improvement'] for stats in analysis.values()]

    print("\nSummary:")
    if sorted_clusters:
        print(f"  Best performing cluster: {sorted_clusters[0][0]} (Acc: {sorted_clusters[0][1]['final_accuracy']:.4f})")
        print(f"  Worst performing cluster: {sorted_clusters[-1][0]} (Acc: {sorted_clusters[-1][1]['final_accuracy']:.4f})")
        print(f"  Performance gap: {sorted_clusters[0][1]['final_accuracy'] - sorted_clusters[-1][1]['final_accuracy']:.4f}")

    if final_accs:
        print(f"  Average cluster accuracy: {np.mean(final_accs):.4f} (±{np.std(final_accs):.4f})")
        print(f"  Average improvement: {np.mean(improvements):.4f}")
        print(f"  Clusters that improved: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")

    print("=" * 60)

    total_clients = args.clients
    participating_clients = sum(len(client_ids) for client_ids in clusters.values())
    ignored_clients = total_clients - participating_clients

    print(f"Total clients: {total_clients}")
    print(f"Participating clients: {participating_clients}")
    print(f"Ignored clients (single-client clusters or clusters with 2 clients): {ignored_clients}")
    print(f"Participation rate: {participating_clients/total_clients*100:.1f}%")


    # Analyze cluster composition diversity
    print(f"\nCluster composition analysis:")
    for cluster_id, client_ids in clusters.items():
        if len(client_ids) > 1:
            # Calculate diversity within this cluster
            histograms = []
            for client_id in client_ids:
                _, y_train = clients_data[client_id]
                labels = np.argmax(y_train, axis=1)
                histogram = np.zeros(config['num_classes'])
                unique, counts = np.unique(labels, return_counts=True)
                histogram[unique] = counts
                histogram = histogram / histogram.sum()
                histograms.append(histogram)

            # Calculate pairwise distances within cluster
            distances = []
            for i in range(len(histograms)):
                for j in range(i + 1, len(histograms)):
                    dist = jensen_shannon_distance(histograms[i], histograms[j])
                    distances.append(dist)

            avg_diversity = np.mean(distances) if distances else 0
            print(f"  Anti-Cluster {cluster_id}: {len(client_ids)} clients, diversity: {avg_diversity:.4f}")

    print("=" * 60)