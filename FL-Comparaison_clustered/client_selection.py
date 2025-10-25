"""
Client Selection Strategies for Federated Learning
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy


def build_djs_clusters(clients_data, js_threshold, num_classes, verbose=False):
    """
    Build clusters using Jensen-Shannon divergence on data histograms

    Args:
        clients_data: List of (x_train, y_train) tuples
        js_threshold: Distance threshold for clustering
        num_classes: Number of classes
        verbose: Print verbose output

    Returns:
        clusters: Dict mapping cluster_id to list of client_ids
        histograms: List of normalized histograms for each client
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

    # Step 2: Build Jensen-Shannon distance matrix
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
        print_js_statistics(js_matrix)

    # Step 3: Agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=js_threshold
    )
    cluster_labels = clustering.fit_predict(js_matrix)

    # Step 4: Create clusters dictionary
    clusters = {}
    for client_idx, cluster_id in enumerate(cluster_labels):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(client_idx)

    return clusters, client_histograms


def print_js_statistics(js_matrix):
    """Print Jensen-Shannon distance statistics"""
    js_values = js_matrix[np.triu_indices_from(js_matrix, k=1)]
    print(f"\n  JS Distance Statistics:")
    print(f"    Min: {js_values.min():.4f}, Max: {js_values.max():.4f}")
    print(f"    Mean: {js_values.mean():.4f}, Median: {np.median(js_values):.4f}")


def select_clients_by_size(client_ids, clients_data, selection_ratio):
    """
    Select clients by data size (original method)

    Args:
        client_ids: List of client IDs in the cluster
        clients_data: List of (x_train, y_train) tuples
        selection_ratio: Ratio of clients to select (0-1)

    Returns:
        List of selected client IDs
    """
    if selection_ratio >= 1.0:
        return client_ids

    # Count samples per client
    client_samples = []
    for client_id in client_ids:
        _, y_train = clients_data[client_id]
        num_samples = len(y_train)
        client_samples.append((client_id, num_samples))

    # Sort by sample size (descending)
    client_samples.sort(key=lambda x: x[1], reverse=True)

    # Select top clients
    num_to_select = max(1, int(len(client_ids) * selection_ratio))
    selected = [client_id for client_id, _ in client_samples[:num_to_select]]

    return selected


def select_clients_by_diversity(client_ids, clients_data, selection_ratio, num_classes):
    """
    Select clients by distribution diversity (NEW METHOD)
    Prioritizes clients with diverse data distributions

    Args:
        client_ids: List of client IDs in the cluster
        clients_data: List of (x_train, y_train) tuples
        selection_ratio: Ratio of clients to select (0-1)
        num_classes: Number of classes

    Returns:
        List of selected client IDs
    """
    if selection_ratio >= 1.0:
        return client_ids

    if len(client_ids) == 1:
        return client_ids

    # Calculate histograms for each client
    client_histograms = {}
    for client_id in client_ids:
        _, y_train = clients_data[client_id]
        labels = np.argmax(y_train, axis=1)

        histogram = np.zeros(num_classes)
        unique, counts = np.unique(labels, return_counts=True)
        histogram[unique] = counts
        histogram = histogram / histogram.sum()  # Normalize

        client_histograms[client_id] = histogram

    # Calculate diversity scores
    diversity_scores = calculate_diversity_scores(client_histograms, client_ids)

    # Select clients with highest diversity scores
    num_to_select = max(1, int(len(client_ids) * selection_ratio))
    selected_clients = sorted(diversity_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [client_id for client_id, _ in selected_clients[:num_to_select]]

    return selected


def calculate_diversity_scores(client_histograms, client_ids):
    """
    Calculate diversity score for each client based on their distance to others

    Higher score = more diverse/different from others

    Args:
        client_histograms: Dict mapping client_id to histogram
        client_ids: List of client IDs

    Returns:
        Dict mapping client_id to diversity score
    """
    n_clients = len(client_ids)

    if n_clients == 1:
        return {client_ids[0]: 1.0}

    # Build distance matrix using JS divergence
    histograms_list = [client_histograms[cid] for cid in client_ids]

    # Calculate pairwise JS distances
    distances = np.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            js_dist = jensen_shannon_distance(histograms_list[i], histograms_list[j])
            distances[i, j] = distances[j, i] = js_dist

    # Diversity score = average distance to all other clients
    diversity_scores = {}
    for idx, client_id in enumerate(client_ids):
        # Average distance to all other clients
        avg_distance = np.mean(distances[idx, :])
        diversity_scores[client_id] = avg_distance

    return diversity_scores


def select_clients_balanced(client_ids, clients_data, selection_ratio, num_classes, alpha=0.5):
    """
    Balanced selection: considers both size AND diversity

    Args:
        client_ids: List of client IDs in the cluster
        clients_data: List of (x_train, y_train) tuples
        selection_ratio: Ratio of clients to select (0-1)
        num_classes: Number of classes
        alpha: Balance parameter (0=only size, 1=only diversity, 0.5=balanced)

    Returns:
        List of selected client IDs
    """
    if selection_ratio >= 1.0:
        return client_ids

    if len(client_ids) == 1:
        return client_ids

    # Calculate size scores
    size_scores = {}
    for client_id in client_ids:
        _, y_train = clients_data[client_id]
        size_scores[client_id] = len(y_train)

    # Normalize size scores to [0, 1]
    max_size = max(size_scores.values())
    min_size = min(size_scores.values())
    if max_size > min_size:
        size_scores = {cid: (size - min_size) / (max_size - min_size)
                       for cid, size in size_scores.items()}
    else:
        size_scores = {cid: 1.0 for cid in client_ids}

    # Calculate diversity scores
    client_histograms = {}
    for client_id in client_ids:
        _, y_train = clients_data[client_id]
        labels = np.argmax(y_train, axis=1)

        histogram = np.zeros(num_classes)
        unique, counts = np.unique(labels, return_counts=True)
        histogram[unique] = counts
        histogram = histogram / histogram.sum()

        client_histograms[client_id] = histogram

    diversity_scores = calculate_diversity_scores(client_histograms, client_ids)

    # Normalize diversity scores to [0, 1]
    max_div = max(diversity_scores.values())
    min_div = min(diversity_scores.values())
    if max_div > min_div:
        diversity_scores = {cid: (div - min_div) / (max_div - min_div)
                           for cid, div in diversity_scores.items()}
    else:
        diversity_scores = {cid: 1.0 for cid in client_ids}

    # Combined score: alpha * diversity + (1-alpha) * size
    combined_scores = {}
    for client_id in client_ids:
        combined_scores[client_id] = (
            alpha * diversity_scores[client_id] +
            (1 - alpha) * size_scores[client_id]
        )

    # Select clients with highest combined scores
    num_to_select = max(1, int(len(client_ids) * selection_ratio))
    selected_clients = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [client_id for client_id, _ in selected_clients[:num_to_select]]

    return selected


def select_clients_random(client_ids, selection_ratio):
    """
    Random selection of clients

    Args:
        client_ids: List of client IDs
        selection_ratio: Ratio of clients to select (0-1)

    Returns:
        List of selected client IDs
    """
    if selection_ratio >= 1.0:
        return client_ids

    num_to_select = max(1, int(len(client_ids) * selection_ratio))
    selected = np.random.choice(client_ids, size=num_to_select, replace=False).tolist()

    return selected


def jensen_shannon_distance(p, q):
    """Calculate Jensen-Shannon distance between two distributions"""
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    return np.sqrt(js_div)


def select_clients_in_clusters(clusters, clients_data, selection_ratio,
                                selection_mode='size', num_classes=10, alpha=0.5):
    """
    Select subset of clients in each cluster using specified strategy

    Args:
        clusters: Dict mapping cluster_id to list of client_ids
        clients_data: List of (x_train, y_train) tuples
        selection_ratio: Ratio of clients to select (0-1)
        selection_mode: Strategy to use
            - 'size': Select by data size (most data first)
            - 'diversity': Select by distribution diversity
            - 'balanced': Combine size and diversity (controlled by alpha)
            - 'random': Random selection
        num_classes: Number of classes (needed for diversity calculations)
        alpha: Balance parameter for 'balanced' mode (0=size, 1=diversity)

    Returns:
        Dict mapping cluster_id to selected client_ids
    """
    selected_clusters = {}

    for cluster_id, client_ids in clusters.items():
        if selection_mode == 'size':
            selected = select_clients_by_size(client_ids, clients_data, selection_ratio)

        elif selection_mode == 'diversity':
            selected = select_clients_by_diversity(client_ids, clients_data,
                                                   selection_ratio, num_classes)

        elif selection_mode == 'balanced':
            selected = select_clients_balanced(client_ids, clients_data,
                                              selection_ratio, num_classes, alpha)

        elif selection_mode == 'random':
            selected = select_clients_random(client_ids, selection_ratio)

        else:
            raise ValueError(f"Unknown selection mode: {selection_mode}")

        selected_clusters[cluster_id] = selected

    return selected_clusters