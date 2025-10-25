
import time
import numpy as np
from models import initialize_global_model, initialize_edge_models
from hierarchical_server import get_cyclic_partner
from utils import save_client_stats_csv
from config import SAVE_CLIENTS_STATS, VERBOSE

def train_vanilla_fl(clients, test_data, args):
    """Entraînement FL vanilla (FedAvg classique)"""
    print(" Train of Vanilla FL (FedAvg)...")

    global_model = initialize_global_model(args.dataset)

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
        global_weights = fedavg_aggregate(client_updates, client_sizes)
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


def train_cyclic_agglomerative(clients, test_data, edge_servers, hierarchical_server, args):
    """
    Cyclic-Agglomerative training with dynamic edge server aggregation
    At each round, edge servers aggregate their clients + a cyclic partner edge server
    """
    print(f" Train of FL Cyclic-Agglomerative...")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    accuracy_history = []
    communication_costs = []
    round_times = []

    num_edges = len(edge_servers)

    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0

        # ÉTAPE 1: Diffuser le modèle global vers les edge servers
        for edge_server in edge_servers:
            edge_server.set_global_model(global_model.get_weights())
            communication_cost += 1  # Global -> Edge

        # ÉTAPE 2: Chaque edge server entraîne ses clients
        edge_client_aggregates = {}  # Store aggregated client models per edge
        edge_data_sizes = {}

        for edge_server in edge_servers:
            edge_clients = [client for client in clients if client.client_id in edge_server.client_ids]

            if not edge_clients:
                continue

            # Entraînement local des clients
            client_updates = []
            client_data_sizes = []

            for client in edge_clients:
                client.update_model(edge_server.local_model)
                client.train_local(args.local_epochs)
                communication_cost += 2  # Edge -> Client -> Edge

                client_updates.append(client.local_model.get_weights())
                client_data_sizes.append(len(client.x_train))

            # Agrégation des clients de cet edge
            if client_updates:
                edge_client_aggregates[edge_server.edge_id] = fedavg_aggregate(client_updates, client_data_sizes)
                edge_data_sizes[edge_server.edge_id] = sum(client_data_sizes)

        # ÉTAPE 3: Agrégation cyclique entre edge servers
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            edge_id = edge_server.edge_id

            if edge_id not in edge_client_aggregates:
                continue

            # Get cyclic partner
            partner_id = get_cyclic_partner(edge_id, round_num, num_edges)

            # Prepare models and weights for aggregation
            models_to_aggregate = [edge_client_aggregates[edge_id]]
            weights_to_aggregate = [edge_data_sizes[edge_id]]

            # Add partner's model if available
            if partner_id in edge_client_aggregates and partner_id != edge_id:
                models_to_aggregate.append(edge_client_aggregates[partner_id])
                weights_to_aggregate.append(edge_data_sizes[partner_id])
                communication_cost += 1  # Edge-to-edge communication

                if VERBOSE:
                    print(f"    Edge {edge_id} aggregating with partner Edge {partner_id}")

            # Aggregate current edge + cyclic partner
            edge_aggregated = fedavg_aggregate(models_to_aggregate, weights_to_aggregate)
            edge_server.local_model.set_weights(edge_aggregated)

            edge_updates.append(edge_aggregated)
            edge_weights.append(sum(weights_to_aggregate))

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
        'method': 'cyclic-agglomerative',
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'round_times': round_times
    }


def train_improved_agglomerative(clients, test_data, edge_servers, hierarchical_server, args):
    """
    Improved Agglomerative training with:
    - Dynamic random client selection per round
    - Alternating aggregation policy (intra-cluster vs global)
    - Periodic global aggregation (e.g., every 5 rounds)

    Args:
        clients: List of FederatedClient objects
        test_data: Test dataset tuple (x_test, y_test)
        edge_servers: List of EdgeServer objects from agglomerative clustering
        hierarchical_server: HierarchicalServer object
        args: Training arguments
    """
    print(f"Train of FL Improved-Agglomerative...")
    print(f"  - Dynamic client selection ratio: {args.selection_variance}")
    print(f"  - Global aggregation every {args.global_aggregation_interval} rounds")
    print(f"  - Alternating aggregation policy enabled\n")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    accuracy_history = []
    communication_costs = []
    round_times = []
    aggregation_history = []  # Track aggregation type per round

    # Configuration for improved method
    GLOBAL_AGG_INTERVAL = getattr(args, 'global_aggregation_interval', 5)
    BASE_SELECTION_RATIO = 0.5
    SELECTION_VARIANCE = getattr(args, 'selection_variance', 0.2)  # ±20% variance

    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0

        # STEP 1: Dynamic client selection ratio (random variation)
        # Randomly vary the selection ratio within bounds
        min_ratio = max(0.1, BASE_SELECTION_RATIO - SELECTION_VARIANCE)
        max_ratio = min(1.0, BASE_SELECTION_RATIO + SELECTION_VARIANCE)
        current_selection_ratio = np.random.uniform(min_ratio, max_ratio)

        if VERBOSE:
            print(f"  Client selection ratio this round: {current_selection_ratio:.2f}")

        # STEP 2: Determine aggregation policy for this round
        # Alternating policy: some rounds favor intra-cluster, others inter-cluster
        is_global_round = (round_num + 1) % GLOBAL_AGG_INTERVAL == 0
        is_intra_cluster_focus = (round_num % 2 == 0) and not is_global_round

        aggregation_type = "global" if is_global_round else ("intra-cluster" if is_intra_cluster_focus else "mixed")
        aggregation_history.append(aggregation_type)

        if VERBOSE:
            print(f"  Aggregation type: {aggregation_type}")

        # STEP 3: Broadcast global model to edge servers
        for edge_server in edge_servers:
            edge_server.set_global_model(global_model.get_weights())
            communication_cost += 1  # Global -> Edge

        # STEP 4: Each edge server trains selected clients
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            # Get all clients assigned to this edge server
            edge_clients = [client for client in clients
                            if client.client_id in edge_server.client_ids]

            if not edge_clients:
                continue

            # Dynamic client selection based on current ratio
            num_clients_to_select = max(1, int(len(edge_clients) * current_selection_ratio))

            # Random selection of clients (instead of always selecting top by data size)
            # This introduces stochasticity and helps avoid overfitting to large-data clients
            if args.random_client_selection:
                selected_clients = np.random.choice(edge_clients,
                                                    size=num_clients_to_select,
                                                    replace=False)
            else:
                # Original selection by data size for comparison
                client_data_sizes = [(client, len(client.x_train)) for client in edge_clients]
                client_data_sizes.sort(key=lambda x: x[1], reverse=True)
                selected_clients = [client for client, _ in client_data_sizes[:num_clients_to_select]]

            # Train selected clients
            client_updates = []
            client_data_sizes = []

            for client in selected_clients:
                client.update_model(edge_server.local_model)
                client.train_local(args.local_epochs)
                communication_cost += 2  # Edge -> Client -> Edge

                client_updates.append(client.local_model.get_weights())
                client_data_sizes.append(len(client.x_train))

            # STEP 5: Edge-level aggregation based on policy
            if client_updates:
                if is_intra_cluster_focus:
                    # Intra-cluster focus: heavier weight on local aggregation
                    edge_aggregated = fedavg_aggregate(client_updates, client_data_sizes)
                    edge_server.local_model.set_weights(edge_aggregated)

                    # Keep more of the edge model's specialization
                    edge_updates.append(edge_aggregated)
                    edge_weights.append(sum(client_data_sizes) * 1.5)  # Boost weight
                else:
                    # Standard aggregation
                    edge_aggregated = fedavg_aggregate(client_updates, client_data_sizes)
                    edge_server.local_model.set_weights(edge_aggregated)
                    edge_updates.append(edge_aggregated)
                    edge_weights.append(sum(client_data_sizes))

        # STEP 6: Global aggregation based on policy
        if is_global_round and edge_updates:
            # Full global aggregation
            global_aggregated = fedavg_aggregate(edge_updates, edge_weights)
            global_model.set_weights(global_aggregated)

            # Update all edge servers with new global model
            for edge_server in edge_servers:
                edge_server.local_model.set_weights(global_aggregated)

            communication_cost += len(edge_servers) * 2  # Bidirectional

            if VERBOSE:
                print(f"  Global aggregation performed!")

        elif not is_intra_cluster_focus and edge_updates:
            # Partial global aggregation (mixed mode)
            # Global model gets updated but edge servers maintain some independence
            global_aggregated = fedavg_aggregate(edge_updates, edge_weights)

            # Blend global and edge models (momentum-like update)
            blend_factor = 0.7  # 70% global, 30% local
            for edge_server in edge_servers:
                edge_weights_current = edge_server.local_model.get_weights()
                blended_weights = [
                    blend_factor * gw + (1 - blend_factor) * ew
                    for gw, ew in zip(global_aggregated, edge_weights_current)
                ]
                edge_server.local_model.set_weights(blended_weights)

            global_model.set_weights(global_aggregated)
            communication_cost += len(edge_servers)

        # STEP 7: Evaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)

        round_time = time.time() - round_start_time

        accuracy_history.append(test_acc)
        communication_costs.append(communication_cost)
        round_times.append(round_time)

        print(f"  Accuracy: {test_acc:.4f}, Time: {round_time:.2f}s, Comm: {communication_cost}")

    # Save extended statistics
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return {
        'method': 'improved-agglomerative',
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'round_times': round_times,
        'aggregation_history': aggregation_history,
        'config': {
            'base_selection_ratio': BASE_SELECTION_RATIO,
            'selection_variance': SELECTION_VARIANCE,
            'global_agg_interval': GLOBAL_AGG_INTERVAL,
            'random_selection': args.random_client_selection
        }
    }


def train_adaptive_agglomerative(clients, test_data, edge_servers, hierarchical_server, args):
    """
    Advanced version with adaptive selection ratio based on convergence
    """
    print(f"Train of FL Adaptive-Agglomerative...")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    accuracy_history = []
    communication_costs = []
    round_times = []
    selection_ratios = []

    # Adaptive parameters
    GLOBAL_AGG_INTERVAL = getattr(args, 'global_aggregation_interval', 5)
    base_selection_ratio = 0.5
    min_improvement_threshold = 0.001  # Minimum accuracy improvement to maintain ratio

    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0

        # Adaptive selection ratio based on convergence
        if round_num > 5 and len(accuracy_history) > 5:
            # Check recent improvement
            recent_improvement = accuracy_history[-1] - accuracy_history[-5]

            if recent_improvement < min_improvement_threshold:
                # Increase diversity by selecting more clients
                current_selection_ratio = min(1.0, base_selection_ratio * 1.5)
            else:
                # Good progress, can be more selective
                current_selection_ratio = max(0.2, base_selection_ratio * 0.8)
        else:
            # Initial rounds: use base ratio with small random variation
            current_selection_ratio = base_selection_ratio * np.random.uniform(0.8, 1.2)

        selection_ratios.append(current_selection_ratio)

        # Determine if this is a global aggregation round
        is_global_round = (round_num + 1) % GLOBAL_AGG_INTERVAL == 0

        # Broadcast model to edge servers
        for edge_server in edge_servers:
            edge_server.set_global_model(global_model.get_weights())
            communication_cost += 1

        # Edge server operations
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            edge_clients = [client for client in clients
                            if client.client_id in edge_server.client_ids]

            if not edge_clients:
                continue

            # Adaptive client selection
            num_to_select = max(1, int(len(edge_clients) * current_selection_ratio))

            # Mixed selection strategy: combine random and performance-based
            if round_num % 3 == 0:
                # Every 3rd round: purely random selection for diversity
                selected_clients = list(np.random.choice(edge_clients,
                                                         size=num_to_select,
                                                         replace=False))
            else:
                # Performance-weighted random selection
                client_scores = []
                for client in edge_clients:
                    # Score based on data size and past performance
                    data_size_score = float(len(client.x_train)) / 1000.0
                    performance_score = float(np.mean(getattr(client, 'avg_local_accuracy', 0.5)))
                    combined_score = 0.7 * data_size_score + 0.3 * performance_score
                    client_scores.append(combined_score)

                # Normalize scores to probabilities
                scores_array = np.array(client_scores)
                probabilities = scores_array / scores_array.sum()

                selected_clients = list(np.random.choice(edge_clients,
                                                         size=num_to_select,
                                                         replace=False,
                                                         p=probabilities))

            # Train selected clients
            client_updates = []
            client_weights = []

            for client in selected_clients:
                client.update_model(edge_server.local_model)

                # Local training with evaluation
                trained_model = client.train_local(args.local_epochs)
                communication_cost += 2

                # Track client performance (for adaptive selection)
                local_acc = client.evaluate_local()[1]
                if hasattr(client, 'avg_local_accuracy'):
                    client.avg_local_accuracy = 0.9 * np.mean(client.avg_local_accuracy) + 0.1 * local_acc
                else:
                    client.avg_local_accuracy = local_acc

                client_updates.append(trained_model.get_weights())
                client_weights.append(len(client.x_train))

            if client_updates:
                edge_aggregated = fedavg_aggregate(client_updates, client_weights)
                edge_server.local_model.set_weights(edge_aggregated)
                edge_updates.append(edge_aggregated)
                edge_weights.append(sum(client_weights))

        # Global aggregation logic
        if edge_updates:
            if is_global_round:
                # Full global synchronization
                global_aggregated = fedavg_aggregate(edge_updates, edge_weights)
                global_model.set_weights(global_aggregated)

                # Sync all edges
                for edge_server in edge_servers:
                    edge_server.local_model.set_weights(global_aggregated)
                communication_cost += len(edge_servers) * 2
            else:
                # Asynchronous partial updates
                global_aggregated = fedavg_aggregate(edge_updates, edge_weights)

                # Momentum-based update
                momentum = 0.9
                current_global = global_model.get_weights()
                updated_global = [
                    momentum * cw + (1 - momentum) * nw
                    for cw, nw in zip(current_global, global_aggregated)
                ]
                global_model.set_weights(updated_global)
                communication_cost += len(edge_servers)

        # Evaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)

        round_time = time.time() - round_start_time

        accuracy_history.append(test_acc)
        communication_costs.append(communication_cost)
        round_times.append(round_time)

        print(f"  Accuracy: {test_acc:.4f}, Time: {round_time:.2f}s, "
              f"Comm: {communication_cost}, Selection: {current_selection_ratio:.2f}")

    return {
        'method': 'adaptive-agglomerative',
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'round_times': round_times,
        'selection_ratios': selection_ratios
    }


def train_random_clustered(clients, test_data, edge_servers, hierarchical_server, args):
    """
    Entraînement Random Clustered FL
    - Clustering fixe basé sur la similarité des distributions
    - Sélection ALÉATOIRE des clients intra-cluster à CHAQUE round
    - Agrégation au serveur central tous les N rounds

    Args:
        clients: Liste des clients
        test_data: Données de test
        edge_servers: Serveurs edge (clusters)
        hierarchical_server: Serveur hiérarchique
        args: Arguments de configuration
    """
    print(f" Train of Random Clustered FL...")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    accuracy_history = []
    communication_costs = []
    round_times = []
    edge_comm_costs = []
    global_comm_costs = []

    # Fréquence d'agrégation au serveur central
    global_agg_freq = getattr(args, 'global_aggregation_interval', 5)
    selection_ratio = args.client_selection_ratio

    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0
        edge_comm = 0
        global_comm = 0

        # ÉTAPE 1: Diffuser le modèle depuis les edge servers vers les clients sélectionnés
        # (ou depuis le global si c'est un round d'agrégation globale)
        should_aggregate_global = (round_num + 1) % global_agg_freq == 0

        if should_aggregate_global or round_num == 0:
            # Diffuser le modèle global vers tous les edge servers
            for edge_server in edge_servers:
                edge_server.set_global_model(global_model.get_weights())
                global_comm += 1  # Global -> Edge

        # ÉTAPE 2: Sélection ALÉATOIRE des clients par cluster et entraînement local
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            # SÉLECTION ALÉATOIRE des clients du cluster
            all_cluster_clients = edge_server.client_ids
            num_to_select = max(1, int(len(all_cluster_clients) * selection_ratio))

            # Sélection aléatoire (change à chaque round!)
            selected_client_ids = np.random.choice(
                all_cluster_clients,
                size=num_to_select,
                replace=False
            ).tolist()

            if VERBOSE:
                print(
                    f"  Cluster {edge_server.edge_id}: Selected {len(selected_client_ids)}/{len(all_cluster_clients)} clients")
                print(f"    Selected IDs: {selected_client_ids}")

            # Récupérer les clients sélectionnés
            edge_clients = [client for client in clients
                            if client.client_id in selected_client_ids]

            if not edge_clients:
                continue

            # Entraînement local des clients sélectionnés
            client_updates = []
            client_data_sizes = []

            for client in edge_clients:
                client.update_model(edge_server.local_model)
                client.train_local(args.local_epochs)
                edge_comm += 2  # Edge -> Client -> Edge

                client_updates.append(client.local_model.get_weights())
                client_data_sizes.append(len(client.x_train))

            # ÉTAPE 3: Agrégation intra-cluster
            if client_updates:
                cluster_aggregated = fedavg_aggregate(client_updates, client_data_sizes)
                edge_server.local_model.set_weights(cluster_aggregated)

                edge_updates.append(cluster_aggregated)
                edge_weights.append(sum(client_data_sizes))

        # ÉTAPE 4: Agrégation au serveur CENTRAL (tous les N rounds)
        if should_aggregate_global:
            if edge_updates:
                global_aggregated = fedavg_aggregate(edge_updates, edge_weights)
                global_model.set_weights(global_aggregated)

                # Communication Edge -> Global
                global_comm += len(edge_servers)

            if VERBOSE:
                print(f"  ✓ Global aggregation performed at round {round_num + 1}")

        # ÉTAPE 5: Évaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)

        # Métriques
        round_time = time.time() - round_start_time
        communication_cost = edge_comm + global_comm

        accuracy_history.append(test_acc)
        communication_costs.append(communication_cost)
        edge_comm_costs.append(edge_comm)
        global_comm_costs.append(global_comm)
        round_times.append(round_time)

        print(
            f"  Accuracy: {test_acc:.4f}, Time: {round_time:.2f}s, Comm: {communication_cost} (Edge: {edge_comm}, Global: {global_comm})")

    # Sauvegarde des statistiques clients
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return {
        'method': 'random-clustered',
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'edge_comm_costs': edge_comm_costs,
        'global_comm_costs': global_comm_costs,
        'total_comm_costs': [e + g for e, g in zip(edge_comm_costs, global_comm_costs)],
        'round_times': round_times
    }


def train_random_clustered_ucb(clients, test_data, edge_servers,
                               hierarchical_server, args):
    """
    Training avec UCB + Model Divergence
    """
    import time
    import numpy as np
    from models import initialize_global_model, initialize_edge_models
    from ucb import UCBDivergenceSelector, compute_js_divergence, compute_coverage_rare_classes
    from utils import save_client_stats_csv
    from config import SAVE_CLIENTS_STATS, VERBOSE

    print(f" Train of Random Clustered FL with UCB Divergence...")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    accuracy_history = []
    communication_costs = []
    round_times = []
    edge_comm_costs = []
    global_comm_costs = []

    global_agg_freq = getattr(args, 'global_aggregation_interval', 5)

    # ✨ Initialiser UCB selector pour chaque edge server
    for edge_server in edge_servers:
        edge_server.ucb_selector = UCBDivergenceSelector(
            num_clients=len(edge_server.client_ids),
            ucb_c=2.0,  # Plus élevé pour Non-IID
            freshness_lambda=0.2,
            reward_window=10
        )

        # Calculer distribution labels du cluster
        from data_preparation import number_classes
        cluster_label_counts = np.zeros(number_classes(args.dataset))
        for client_id in edge_server.client_ids:
            client = clients[client_id]
            client_labels = np.argmax(client.y_train, axis=1)
            unique, counts = np.unique(client_labels, return_counts=True)
            cluster_label_counts[unique] += counts

        edge_server.cluster_label_distribution = cluster_label_counts

    # Training loop
    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0
        edge_comm = 0
        global_comm = 0

        should_aggregate_global = (round_num + 1) % global_agg_freq == 0

        # Diffusion globale si nécessaire
        if should_aggregate_global or round_num == 0:
            for edge_server in edge_servers:
                edge_server.set_global_model(global_model.get_weights())
                global_comm += 1

        # ✨ SÉLECTION UCB ET ENTRAÎNEMENT PAR EDGE SERVER
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            selector = edge_server.ucb_selector

            # [1] Sélection UCB
            k = max(1, int(len(edge_server.client_ids) * args.client_selection_ratio))
            selected_local_ids = selector.select_clients(
                available_clients=list(range(len(edge_server.client_ids))),
                k=k,
                round_t=round_num
            )

            # Mapper vers IDs globaux
            selected_global_ids = [edge_server.client_ids[i] for i in selected_local_ids]

            if VERBOSE:
                print(f"  Cluster {edge_server.edge_id}: Selected {selected_global_ids}")

            # [2] Entraînement clients sélectionnés
            client_updates = []
            client_data_sizes = []
            cluster_weights_before = edge_server.local_model.get_weights()

            for local_id, global_id in zip(selected_local_ids, selected_global_ids):
                client = clients[global_id]

                # Entraînement
                client.update_model(edge_server.local_model)
                client.train_local(args.local_epochs)
                edge_comm += 2  # Edge -> Client -> Edge

                client_weights = client.local_model.get_weights()
                client_updates.append(client_weights)
                client_data_sizes.append(len(client.x_train))

                # [3] Calculer update vector
                client_update_vector = flatten_weights(client_weights) - flatten_weights(cluster_weights_before)

                # [4] Métriques diversité
                client_label_dist = get_client_label_distribution(client, args.dataset)

                diversity_js = compute_js_divergence(
                    client_label_dist,
                    edge_server.cluster_label_distribution
                )

                coverage_rare = compute_coverage_rare_classes(
                    client_label_dist,
                    edge_server.cluster_label_distribution,
                    rare_threshold=0.05
                )

                # [5] Calculer reward
                reward = selector.compute_reward(
                    client_id=local_id,
                    client_update=client_update_vector,
                    diversity_js=diversity_js,
                    coverage_rare=coverage_rare,
                    round_t=round_num,
                    total_rounds=args.rounds
                )

                # [6] Update consensus
                selector.update_after_training(
                    client_id=local_id,
                    client_update=client_update_vector,
                    reward=reward
                )

                if VERBOSE:
                    print(f"    Client {global_id}: reward={reward:.3f}, "
                          f"div={diversity_js:.3f}, cov={coverage_rare:.3f}")

            # [7] Agrégation intra-cluster
            if client_updates:
                cluster_aggregated = fedavg_aggregate(client_updates, client_data_sizes)
                edge_server.local_model.set_weights(cluster_aggregated)

                edge_updates.append(cluster_aggregated)
                edge_weights.append(sum(client_data_sizes))

        # [8] Agrégation globale périodique
        if should_aggregate_global:
            if edge_updates:
                global_aggregated = fedavg_aggregate(edge_updates, edge_weights)
                global_model.set_weights(global_aggregated)
                global_comm += len(edge_servers)

            # Reset consensus de tous les edge servers
            for edge_server in edge_servers:
                edge_server.ucb_selector.reset_for_global_aggregation()

            if VERBOSE:
                print(f"  ✓ Global aggregation performed at round {round_num + 1}")

        # [9] Évaluation
        x_test, y_test = test_data
        y_test_labels = np.argmax(y_test, axis=1)
        test_loss, test_acc = global_model.evaluate(x_test, y_test_labels, verbose=0)

        # Métriques
        round_time = time.time() - round_start_time
        communication_cost = edge_comm + global_comm

        accuracy_history.append(test_acc)
        communication_costs.append(communication_cost)
        edge_comm_costs.append(edge_comm)
        global_comm_costs.append(global_comm)
        round_times.append(round_time)

        print(f"  Accuracy: {test_acc:.4f}, Time: {round_time:.2f}s, "
              f"Comm: {communication_cost} (Edge: {edge_comm}, Global: {global_comm})")

        # [10] Log statistiques UCB périodiquement
        if (round_num + 1) % 5 == 0:
            for edge_server in edge_servers:
                stats = edge_server.ucb_selector.get_statistics()
                print(f"  Cluster {edge_server.edge_id} stats:")
                print(f"    Consensus magnitude: {stats['consensus_magnitude']:.3f}")
                sel_counts = stats['selection_counts']
                print(f"    Selection range: [{min(sel_counts.values())}, {max(sel_counts.values())}]")

    # Sauvegarde finale
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return {
        'method': 'random-clustered-ucb',
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'edge_comm_costs': edge_comm_costs,
        'global_comm_costs': global_comm_costs,
        'total_comm_costs': [e + g for e, g in zip(edge_comm_costs, global_comm_costs)],
        'round_times': round_times
    }

def train_vanilla_fl_ucb(clients, test_data, args):
    """
    Entraînement FL vanilla (FedAvg) avec UCB + Model Divergence
    """
    import time
    import numpy as np
    from models import initialize_global_model
    from ucb import VanillaUCBSelector, flatten_weights, get_client_label_distribution
    from utils import save_client_stats_csv
    from config import SAVE_CLIENTS_STATS, VERBOSE

    print(" Training Vanilla FL with UCB Divergence...")

    # Initialiser modèle global
    global_model = initialize_global_model(args.dataset)

    # ✨ Initialiser UCB Selector
    ucb_selector = VanillaUCBSelector(
        num_clients=len(clients),
        ucb_c=getattr(args, 'ucb_c', 2.0),  # Paramètre configurable
        freshness_lambda=getattr(args, 'freshness_lambda', 0.2),
        reward_window=10
    )

    # Initialiser distribution globale des labels
    clients_data = [(c.x_train, c.y_train) for c in clients]
    ucb_selector.initialize_global_distribution(clients_data)

    # Résultats
    results = {
        'accuracy_history': [],
        'communication_costs': [],
        'round_times': []
    }

    # Training loop
    for round_num in range(args.rounds):
        start_time = time.time()

        if VERBOSE:
            print(f"  Round {round_num + 1}/{args.rounds}")

        # ✨ [1] Démarrer nouveau round dans consensus tracker
        ucb_selector.start_new_round()

        # ✨ [2] Sélection UCB des clients
        selection_ratio = getattr(args, 'client_selection_ratio', 0.3)
        k = max(1, int(len(clients) * selection_ratio))

        selected_ids = ucb_selector.select_clients(
            available_clients=list(range(len(clients))),
            k=k,
            round_t=round_num
        )

        if VERBOSE:
            print(f"    Selected {len(selected_ids)}/{len(clients)} clients: {selected_ids}")

        # [3] Entraînement local des clients sélectionnés
        client_updates = []
        client_sizes = []
        global_weights_before = global_model.get_weights()

        for client_id in selected_ids:
            client = clients[client_id]

            # Entraînement local
            client.update_model(global_model)
            client.train_local(args.local_epochs)

            # Récupérer poids
            client_weights = client.local_model.get_weights()
            client_updates.append(client_weights)
            client_sizes.append(len(client.x_train))

            # ✨ [4] Calculer update vector (W_client - W_global)
            client_update_vector = flatten_weights(client_weights) - flatten_weights(global_weights_before)

            # ✨ [5] Distribution labels du client
            client_label_dist = get_client_label_distribution(client, args.dataset)

            # ✨ [6] Calculer reward multi-critères
            reward = ucb_selector.compute_reward(
                client_id=client_id,
                client_update=client_update_vector,
                client_label_dist=client_label_dist,
                round_t=round_num,
                total_rounds=args.rounds
            )

            # ✨ [7] Update consensus avec ce client
            ucb_selector.update_after_training(
                client_id=client_id,
                client_update=client_update_vector,
                reward=reward
            )

            if VERBOSE:
                print(f"      Client {client_id}: reward={reward:.3f}")

        # ✨ [8] Finaliser round (calculer consensus du round)
        ucb_selector.finalize_round()

        # [9] Agrégation FedAvg globale
        global_weights = fedavg_aggregate(client_updates, client_sizes)
        global_model.set_weights(global_weights)

        # [10] Évaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        _, accuracy = global_model.evaluate(x_test, y_test, verbose=0)

        # [11] Métriques
        comm_cost = len(selected_ids) * 2  # Upload + download pour chaque client
        round_time = time.time() - start_time

        results['accuracy_history'].append(accuracy)
        results['communication_costs'].append(comm_cost)
        results['round_times'].append(round_time)

        if VERBOSE:
            print(f"    Accuracy: {accuracy:.4f}, Comm Cost: {comm_cost}, Time: {round_time:.2f}s")

        # ✨ [12] Log statistiques UCB périodiquement
        if (round_num + 1) % 5 == 0 or round_num == args.rounds - 1:
            stats = ucb_selector.get_statistics()
            print(f"\n  📊 UCB Statistics (Round {round_num + 1}):")
            print(f"    Consensus magnitude: {stats['consensus_magnitude']:.3f}")

            sel_counts = stats['selection_counts']
            print(f"    Selection counts: min={min(sel_counts.values())}, "
                  f"max={max(sel_counts.values())}, "
                  f"mean={np.mean(list(sel_counts.values())):.1f}")

            qualities = stats['client_qualities']
            print(f"    Client quality: min={min(qualities.values()):.3f}, "
                  f"max={max(qualities.values()):.3f}, "
                  f"mean={np.mean(list(qualities.values())):.3f}")

            # Calculer Gini coefficient pour équité
            gini = compute_gini_coefficient(list(sel_counts.values()))
            print(f"    Selection equity (Gini): {gini:.3f} (0=perfect equity, 1=total inequality)")

    # [13] Sauvegarde des statistiques clients
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    # [14] Analyse finale détaillée
    print(f"\n{'=' * 60}")
    print("FINAL UCB ANALYSIS")
    print(f"{'=' * 60}")

    final_stats = ucb_selector.get_statistics()
    sel_counts = final_stats['selection_counts']

    print(f"Selection distribution:")
    print(f"  Total selections: {sum(sel_counts.values())}")
    print(f"  Min: {min(sel_counts.values())} (Client {min(sel_counts, key=sel_counts.get)})")
    print(f"  Max: {max(sel_counts.values())} (Client {max(sel_counts, key=sel_counts.get)})")
    print(f"  Mean: {np.mean(list(sel_counts.values())):.2f}")
    print(f"  Std: {np.std(list(sel_counts.values())):.2f}")
    print(f"  Gini: {compute_gini_coefficient(list(sel_counts.values())):.3f}")

    # Identifier clients sous-utilisés
    mean_selections = np.mean(list(sel_counts.values()))
    underused = [c for c, count in sel_counts.items() if count < mean_selections * 0.5]
    if underused:
        print(f"\n⚠️  Underused clients ({len(underused)}): {underused}")
        print(f"  Consider increasing ucb_c or freshness_lambda for more exploration")

    return results


def compute_gini_coefficient(values):
    """
    Calcule coefficient de Gini pour mesurer inégalité de distribution
    0 = parfaite égalité, 1 = inégalité totale
    """
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((n - np.arange(n)) * sorted_values)) / (n * np.sum(sorted_values)) - 1



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

def flatten_weights(weights_list):
    """Aplatit liste de poids en vecteur 1D"""
    return np.concatenate([w.flatten() for w in weights_list])


def get_client_label_distribution(client, dataset_name):
    """Récupère distribution des labels d'un client"""
    from data_preparation import number_classes
    num_classes = number_classes(dataset_name)
    labels = np.argmax(client.y_train, axis=1)
    histogram = np.zeros(num_classes)
    unique, counts = np.unique(labels, return_counts=True)
    histogram[unique] = counts
    return histogram

