"""
Script principal pour comparaison FL Vanilla vs Hierarchical vs Drop-in
Usage: python main.py --hierarchy-type hierarchical --clients 20 --edge-servers 5

Exemples rapides:
  python main.py --hierarchy-type vanilla --clients 20 --epochs 30
  python main.py --hierarchy-type hierarchical --clients 20 --edge-servers 5
  python main.py --hierarchy-type drop-in --clients 20 --edge-servers 5 --epochs 40
  python main.py --hierarchy-type compare --clients 20 --edge-servers 5 --verbose
"""

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

from data_preparation import prepare_federated_cifar10, prepare_federated_mnist
from models import create_model
from hierarchical_server import EdgeServer, HierarchicalServer
from client import FederatedClient
from aggregations.fedavg import FedAvgAggregator
from utils import setup_gpu, save_results, plot_comparison_results, jensen_shannon_distance, select_clients_by_samples


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Comparaison FL Vanilla vs Hierarchical vs Drop-in vs agglomerative',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Types d'entraînement disponibles:

  🔹 Vanilla FL:
     python main.py --hierarchy-type vanilla --clients 20 --epochs 30

  🔹 Hierarchical FL:
     python main.py --hierarchy-type hierarchical --clients 20 --edge-servers 5

  🔹 Drop-in Hierarchical FL:
     python main.py --hierarchy-type drop-in --clients 20 --edge-servers 5

  🔹 Comparaison complète:
     python main.py --hierarchy-type compare --clients 20 --edge-servers 5 --verbose
        """
    )

    # Arguments obligatoires
    parser.add_argument('--hierarchy-type', type=str, required=True,
                        choices=['vanilla', 'hierarchical', 'agglomerative', 'drop-in', 'compare'],
                        help='Type d\'entraînement hiérarchique')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar10'],
                        help='Dataset à utiliser (mnist: 28x28 N&B, cifar: 32x32 couleur)')

    # Arguments optionnels
    parser.add_argument('--clients', type=int, default=20,
                        help='Nombre total de clients (défaut: 20)')

    parser.add_argument('--edge-servers', type=int, default=5,
                        help='Nombre d\'edge servers pour hiérarchie (défaut: 5)')

    parser.add_argument('--js_threshold', type=float, default=0.5,
                        help='Seuil pour clustering agglomératif JS')

    parser.add_argument('--client_selection_ratio', type=float, default=0.3,
                        help='Pourcentage de clients à sélectionner par cluster (ex: 0.3 = 30%)')

    parser.add_argument('--num_classes', type=int, default=10,
                        help='Nombre de classes dans le dataset')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Nombre d\'époques d\'entraînement (défaut: 30)')

    parser.add_argument('--local-epochs', type=int, default=5,
                        help='Époques locales par client (défaut: 5)')

    parser.add_argument('--rounds', type=int, default=20,
                        help='Nombre de rounds de communication (défaut: 20)')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='Taille des lots (défaut: 32)')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Taux d\'apprentissage (défaut: 0.001)')

    parser.add_argument('--iid', action='store_true',
                        help='Distribution IID des données (défaut: non-IID)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU à utiliser (-1 pour CPU, défaut: 0)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire (défaut: 42)')

    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbeux avec détails')

    parser.add_argument('--plot', action='store_true',
                        help='Afficher les graphiques de comparaison')

    return parser.parse_args()


def setup_hierarchy(clients_data, args):
    """Configure la hiérarchie selon le type"""
    total_clients = len(clients_data)

    if args.hierarchy_type == 'vanilla':
        return None, None  # Pas de hiérarchie

    elif args.hierarchy_type == 'hierarchical':
        # Répartition sans chevauchement
        clients_per_edge = total_clients // args.edge_servers
        edge_servers = []

        for i in range(args.edge_servers):
            start_idx = i * clients_per_edge
            end_idx = min(start_idx + clients_per_edge, total_clients)
            client_ids = list(range(start_idx, end_idx))
            edge_servers.append(EdgeServer(i, client_ids))

            if args.verbose:
                print(f"  Edge Server {i}: Clients {client_ids}")

    elif args.hierarchy_type == 'agglomerative':
        # Étape 1: Calculer les histogrammes de labels pour chaque client
        client_histograms = []
        for client_data in clients_data:
            # Extraire les labels du client
            x_train, y_train = client_data
            y_train = np.argmax(y_train, axis=1)
            labels = y_train.flatten() if isinstance(y_train, np.ndarray) else y_train

            # Créer l'histogramme normalisé
            unique, counts = np.unique(labels, return_counts=True)
            histogram = np.zeros(args.num_classes)
            histogram[unique] = counts
            histogram = histogram / histogram.sum()  # Normalisation
            client_histograms.append(histogram)

        # Étape 2: Calculer la matrice de distances Jensen-Shannon (DJS)
        n_clients = len(client_histograms)
        js_matrix = np.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                # Distance JS
                js_dist = jensen_shannon_distance(client_histograms[i], client_histograms[j])
                js_matrix[i, j] = js_dist
                js_matrix[j, i] = js_dist

        # Après la création de js_matrix
        print("\n Analyse des distances JS:")
        js_values = js_matrix[np.triu_indices_from(js_matrix, k=1)]  # Partie supérieure
        print(f"  Distance min: {js_values.min():.4f}")
        print(f"  Distance max: {js_values.max():.4f}")
        print(f"  Distance moyenne: {js_values.mean():.4f}")
        print(f"  Distance médiane: {np.median(js_values):.4f}")
        print(f"  Percentile 25%: {np.percentile(js_values, 25):.4f}")
        print(f"  Percentile 75%: {np.percentile(js_values, 75):.4f}")

        # Étape 3: Clustering agglomératif
        from sklearn.cluster import AgglomerativeClustering
        threshold = args.js_threshold if hasattr(args, 'js_threshold') else 0.3
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=threshold
        )
        cluster_labels = clustering.fit_predict(js_matrix)

        # Étape 4: Créer les edge servers basés sur les clusters
        clusters = {}
        for client_idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client_idx)

        # edge_servers = []
        # for edge_id, client_ids in enumerate(clusters.values()):
        #     edge_servers.append(EdgeServer(edge_id, client_ids))
        #     if args.verbose:
        #         print(f"  Edge Server {edge_id}: Clients {client_ids} (Cluster JS)")
        #
        # if args.verbose:
        #     print(f"  Nombre de clusters formés: {len(clusters)}")
        #     print(f"  Seuil JS utilisé: {threshold}")

        edge_servers = []
        for edge_id, client_ids in enumerate(clusters.values()):
            # Sélectionner les clients avec le plus d'échantillons
            selected_clients = select_clients_by_samples(
                client_ids,
                clients_data,
                args.client_selection_ratio
            )

            edge_servers.append(EdgeServer(edge_id, selected_clients))

            if args.verbose:
                print(f"  Edge Server {edge_id}: {len(selected_clients)}/{len(client_ids)} clients sélectionnés")
                print(f"    Clients actifs: {selected_clients} (Cluster JS)")
        if args.verbose:
            print(f"  Nombre de clusters formés: {len(clusters)}")
            print(f"  Seuil JS utilisé: {threshold}")

    elif args.hierarchy_type == 'drop-in':
        # Drop-in: TOUS les clients distribués + duplications
        clients_per_edge = total_clients // args.edge_servers
        remaining_clients = total_clients % args.edge_servers
        edge_servers = []

        # Étape 1: Distribuer tous les clients de base (sans chevauchement)
        client_idx = 0
        base_assignments = []

        for i in range(args.edge_servers):
            # Nombre de clients de base pour cet edge server
            base_count = clients_per_edge + (1 if i < remaining_clients else 0)
            client_ids = list(range(client_idx, client_idx + base_count))
            base_assignments.append(client_ids)
            client_idx += base_count

        # Étape 2: Ajouter des duplications pour simuler drop-in
        for i in range(args.edge_servers):
            client_ids = base_assignments[i].copy()

            # Ajouter 2-3 clients aléatoires d'autres edge servers
            other_clients = []
            for j in range(args.edge_servers):
                if j != i:  # Clients des autres edge servers
                    other_clients.extend(base_assignments[j])

            if other_clients:
                # Choisir aléatoirement 2-3 clients à dupliquer
                num_duplicates = min(3, len(other_clients))
                duplicates = np.random.choice(
                    other_clients,
                    size=num_duplicates,
                    replace=False
                ).tolist()
                client_ids.extend(duplicates)

            edge_servers.append(EdgeServer(i, client_ids))

            if args.verbose:
                print(f"  Edge Server {i}: Clients {client_ids} (avec drop-in)")

    else:
        # Type de hiérarchie non reconnu
        edge_servers = []
        if args.verbose:
            print(f"⚠️ Type de hiérarchie '{args.hierarchy_type}' non reconnu")

    hierarchical_server = HierarchicalServer(edge_servers)
    return edge_servers, hierarchical_server


def train_vanilla_fl(clients, test_data, args):
    """Entraînement FL vanilla (FedAvg classique)"""
    print("🌐 Entraînement FL Vanilla (FedAvg)...")

    input_shape = (28, 28, 1)
    num_classes = 0

    if args.dataset == 'mnist':
        input_shape = (28, 28, 1)
        num_classes = 10
    elif args.dataset == 'cifar10':
        input_shape = (32, 32, 3)
        num_classes = 10

    # Modèle global
    global_model = create_model(args.dataset, input_shape, num_classes)
    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    aggregator = FedAvgAggregator()

    results = {
        'accuracy_history': [],
        'communication_costs': [],
        'round_times': []
    }

    for round_num in range(args.rounds):
        start_time = time.time()

        if args.verbose:
            print(f"  Round {round_num + 1}/{args.rounds}")

        # Entraînement local
        client_updates = []
        client_sizes = []

        for client in clients:
            client.update_model(global_model)  # Mettre à jour le modèle d'abord
            client.train_local(args.local_epochs)  # Puis entraîner
            weights = client.local_model.get_weights()  # Récupérer les poids
            client_updates.append(weights)
            client_sizes.append(len(client.x_train))

        # Agrégation FedAvg
        global_weights = aggregator.aggregate(client_updates, client_sizes, global_model)
        global_model.set_weights(global_weights)

        # Évaluation
        x_test, y_test = test_data
        y_test_sparse = np.argmax(y_test, axis=1)
        _, accuracy = global_model.evaluate(x_test, y_test_sparse, verbose=0)

        # Métriques
        comm_cost = len(clients) * 2  # Upload + download pour chaque client
        round_time = time.time() - start_time

        results['accuracy_history'].append(accuracy)
        results['communication_costs'].append(comm_cost)
        results['round_times'].append(round_time)

        if args.verbose:
            print(f"    Accuracy: {accuracy:.4f}, Comm Cost: {comm_cost}, Time: {round_time:.2f}s")

    return results


def train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args):
    """Entraînement avec hiérarchie"""
    hierarchy_type = "Drop-in" if args.hierarchy_type == 'drop-in' else "Hierarchical"
    print(f"🏗️ Entraînement FL {hierarchy_type}...")

    # Extraire input_shape et num_classes du test_data (tuple)
    input_shape = (28, 28, 1)
    num_classes = 0
    if args.dataset == 'mnist':
        input_shape = (28, 28, 1)
        num_classes = 10
    elif args.dataset == 'cifar10':
        input_shape = (32, 32, 3)
        num_classes = 10
    global_model = create_model(args.dataset, input_shape, num_classes)

    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # INITIALISER les modèles des edge servers
    for edge_server in edge_servers:
        if edge_server.local_model is None:
            edge_server.local_model = create_model(args.dataset, input_shape, num_classes)
            edge_server.local_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        edge_server.local_model.set_weights(global_model.get_weights())

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
        X_test, y_test = test_data
        y_test_sparse = np.argmax(y_test, axis=1)
        test_loss, test_acc = global_model.evaluate(X_test, y_test_sparse, verbose=0)

        round_time = time.time() - round_start_time

        accuracy_history.append(test_acc)
        communication_costs.append(communication_cost)
        round_times.append(round_time)

        print(f"  Accuracy: {test_acc:.4f}, Temps: {round_time:.2f}s, Comm: {communication_cost}")

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

    for layer_idx in range(len(model_weights_list[0])):
        weighted_layer = sum(
            (data_sizes[i] / total_size) * model_weights_list[i][layer_idx]
            for i in range(len(model_weights_list))
        )
        weights_avg.append(weighted_layer)

    return weights_avg

def compare_all_methods(fed_data, test_data, args):
    """Compare toutes les méthodes"""
    print("🔄 Comparaison de toutes les méthodes...")

    all_results = {}

    # 1. Vanilla FL
    print("\n" + "="*50)
    clients = [FederatedClient(i, data, FedAvgAggregator()) for i, data in enumerate(fed_data)]
    vanilla_results = train_vanilla_fl(clients, test_data, args)
    all_results['vanilla'] = vanilla_results
    save_results({'vanilla': all_results['vanilla']}, args)

    # 2. Hierarchical FL
    print("\n" + "="*50)
    args.hierarchy_type = 'hierarchical'
    edge_servers, hierarchical_server = setup_hierarchy(fed_data, args)
    clients = [FederatedClient(i, data, FedAvgAggregator()) for i, data in enumerate(fed_data)]
    hierarchical_results = train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args)
    all_results['hierarchical'] = hierarchical_results
    save_results({'hierarchical': all_results['hierarchical']}, args)

    # 3. Drop-in FL
    print("\n" + "="*50)
    args.hierarchy_type = 'drop-in'
    edge_servers, hierarchical_server = setup_hierarchy(fed_data, args)
    clients = [FederatedClient(i, data, FedAvgAggregator()) for i, data in enumerate(fed_data)]
    drop_in_results = train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args)
    all_results['drop_in'] = drop_in_results
    save_results({'drop_in': all_results['drop_in']}, args)

    # 4. Agglomerative FL
    print("\n" + "="*50)
    args.hierarchy_type = 'agglomerative'
    edge_servers, hierarchical_server = setup_hierarchy(fed_data, args)
    clients = [FederatedClient(i, data, FedAvgAggregator()) for i, data in enumerate(fed_data)]
    drop_in_results = train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args)
    all_results['agglomerative'] = drop_in_results
    save_results({'agglomerative': all_results['agglomerative']}, args)

    return all_results


def print_final_results(results, hierarchy_type):
    """Affiche les résultats finaux"""
    print("\n" + "="*60)
    print(f"=== RÉSULTATS FINAUX - {hierarchy_type.upper()} ===")
    print("="*60)

    if hierarchy_type == 'compare':
        for method, method_results in results.items():
            final_acc = method_results['accuracy_history'][-1]
            avg_time = np.mean(method_results['round_times'])
            total_comm = sum(method_results.get('total_comm_costs', method_results.get('communication_costs', [])))

            print(f"{method.capitalize()}:")
            print(f"  • Accuracy finale: {final_acc:.4f}")
            print(f"  • Temps moyen/round: {avg_time:.2f}s")
            print(f"  • Coût communication total: {total_comm}")
            print()
    else:
        final_acc = results['accuracy_history'][-1]
        avg_time = np.mean(results['round_times'])

        print(f"Accuracy finale: {final_acc:.4f}")
        print(f"Temps moyen par round: {avg_time:.2f}s")

        if 'total_comm_costs' in results:
            total_comm = sum(results['total_comm_costs'])
            avg_edge_comm = np.mean(results['edge_comm_costs'])
            avg_global_comm = np.mean(results['global_comm_costs'])

            print(f"Coût communication total: {total_comm}")
            print(f"Coût communication edge moyen/round: {avg_edge_comm:.1f}")
            print(f"Coût communication global moyen/round: {avg_global_comm:.1f}")
        else:
            total_comm = sum(results['communication_costs'])
            print(f"Coût communication total: {total_comm}")


def main():
    """Fonction principale"""

    # Parse des arguments
    args = parse_arguments()

    # Configuration initiale
    print("🔄 Démarrage FL Aggregation Hierarchical...")
    print(f"Type: {args.hierarchy_type}")
    print(f"Clients: {args.clients}")
    if args.hierarchy_type != 'vanilla':
        print(f"Edge Servers: {args.edge_servers}")
    print(f"Rounds: {args.rounds}")
    print(f"Distribution: {'IID' if args.iid else 'Non-IID'}")

    # Setup GPU
    setup_gpu(args.gpu)

    # Fixer les graines
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Préparation des données
    print("\n📊 Préparation des données ...")
    if args.dataset == 'cifar10':
        fed_data, test_data, _ = prepare_federated_cifar10(iid=args.iid, num_clients=args.clients)
    elif args.dataset == 'mnist':
        fed_data, test_data, _ = prepare_federated_mnist(iid=args.iid, num_clients=args.clients)


    print(f"✓ {len(fed_data)} clients configurés")
    print(f"✓ Données de test: {len(list(test_data[0]))} échantillons")

    # Exécution selon le type
    if args.hierarchy_type == 'compare':
        # Comparaison de toutes les méthodes
        results = compare_all_methods(fed_data, test_data, args)

        # Sauvegarde et visualisation
        save_results(results, args)
        #if args.plot:
            #plot_comparison_results(results, args)

        print_final_results(results, 'compare')

    elif args.hierarchy_type == 'vanilla':
        # FL Vanilla
        clients = [FederatedClient(i, data, FedAvgAggregator()) for i, data in enumerate(fed_data)]
        results = train_vanilla_fl(clients, test_data, args)

        # Sauvegarde et visualisation
        results['method'] = 'vanilla'
        save_results(results, args)
        #if args.plot:
           # plot_comparison_results({'vanilla': results}, args)

        print_final_results(results, 'vanilla')

    else:
        # FL Hiérarchique ou Drop-in
        edge_servers, hierarchical_server = setup_hierarchy(fed_data, args)
        clients = [FederatedClient(i, data, FedAvgAggregator()) for i, data in enumerate(fed_data)]
        results = train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args)

        # Sauvegarde et visualisation
        results['method'] = args.hierarchy_type
        save_results(results, args)
        #if args.plot:
         #   plot_comparison_results({args.hierarchy_type: results}, args)

        print_final_results(results, args.hierarchy_type)


if __name__ == "__main__":
    main()