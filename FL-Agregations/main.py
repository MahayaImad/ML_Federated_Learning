"""
Point d'entrée principal
"""
import tensorflow as tf
import os


def configure_gpu():
    """Configuration GPU avec gestion stricte de la mémoire"""
    # Limiter la mémoire GPU disponible
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Limiter drastiquement la mémoire GPU utilisable
            tf.config.experimental.set_memory_limit(gpus[0], 1024)  # 1GB seulement
            tf.config.experimental.set_memory_growth(gpus[0], True)

            # Tester avec un tenseur simple
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([1.0])
                _ = test_tensor + 1.0

            print(f"GPU configuré avec limite de 1GB: {gpus[0]}")
            return True

        except Exception as e:
            print(f"GPU saturé, passage forcé au CPU: {e}")
            # FORCER l'utilisation CPU si GPU saturé
            tf.config.set_visible_devices([], 'GPU')
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            return False
    else:
        print("Aucun GPU détecté, utilisation CPU")
        return False


# Configurer avant tout
configure_gpu()
import argparse
from data_preparation import prepare_federated_cifar10
from aggregations import (FedAvgAggregator, FedProxAggregator, ScaffoldAggregator,
                         FedOptAggregator)
from client import FederatedClient
from server import FederatedServer
from experiments.run_comparison import compare_aggregation_methods
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from config import RESULTS_DIR, MODELS_DIR, PLOTS_DIR


def save_comparison_results(results, args):
    """Sauvegarde les résultats de comparaison"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sauvegarde JSON
    results_data = {
        'timestamp': timestamp,
        'configuration': {
            'iid': args.iid,
            'model': args.model,
            'method': args.method
        },
        'results': results
    }

    json_file = os.path.join(RESULTS_DIR, f"comparison_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"Résultats sauvegardés: {json_file}")
    return json_file


def save_single_method_results(server, args, model):
    """Sauvegarde les résultats d'une méthode unique"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.method}_{timestamp}"

    # Sauvegarde avec la méthode du serveur
    json_file = os.path.join(RESULTS_DIR, f"{filename}.json")
    server.save_results(json_file)

    # Sauvegarde du modèle
    model_file = os.path.join(MODELS_DIR, f"{filename}.keras")
    model.save(model_file)

    print(f"Modèle sauvegardé: {model_file}")
    print(f"Métriques sauvegardées: {json_file}")

    return json_file, model_file


def plot_comparison_results(results, args):
    """Génère les graphiques de comparaison"""
    if not results:
        print("Aucun résultat à tracer")
        return

    # Extraire les données
    methods = list(results.keys())
    accuracies = [results[method]['final_accuracy'] for method in methods]
    conv_rates = [results[method]['convergence_rate'] for method in methods]
    comm_costs = [results[method]['communication_cost'] for method in methods]

    # Créer les graphiques
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy finale
    axes[0].bar(methods, accuracies)
    axes[0].set_title('Accuracy Finale par Méthode')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)

    # Taux de convergence
    axes[1].bar(methods, conv_rates)
    axes[1].set_title('Taux de Convergence')
    axes[1].set_ylabel('Taux')
    axes[1].tick_params(axis='x', rotation=45)

    # Coût de communication
    axes[2].bar(methods, comm_costs)
    axes[2].set_title('Coût de Communication')
    axes[2].set_ylabel('Coût')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(PLOTS_DIR, f"comparison_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Graphique sauvegardé: {plot_file}")
    return plot_file


def plot_single_method_metrics(metrics, method_name, args):
    """Graphiques pour une méthode unique"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Accuracy au fil du temps
    if 'test_accuracy' in metrics and metrics['test_accuracy']:
        axes[0, 0].plot(metrics['test_accuracy'], marker='o')
        axes[0, 0].set_title(f'{method_name} - Accuracy Test')
        axes[0, 0].set_xlabel('Tours')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)

    # Temps par tour
    if 'round_times' in metrics and metrics['round_times']:
        axes[0, 1].plot(metrics['round_times'], marker='s')
        axes[0, 1].set_title(f'{method_name} - Temps par Tour')
        axes[0, 1].set_xlabel('Tours')
        axes[0, 1].set_ylabel('Temps (s)')
        axes[0, 1].grid(True)

    # Coûts de communication
    if 'communication_costs' in metrics and metrics['communication_costs']:
        axes[1, 0].plot(metrics['communication_costs'], marker='^')
        axes[1, 0].set_title(f'{method_name} - Coût Communication')
        axes[1, 0].set_xlabel('Tours')
        axes[1, 0].set_ylabel('Coût')
        axes[1, 0].grid(True)

    # Clients participants
    if 'participating_clients' in metrics and metrics['participating_clients']:
        participation = [len(clients) for clients in metrics['participating_clients']]
        axes[1, 1].plot(participation, marker='d')
        axes[1, 1].set_title(f'{method_name} - Clients Participants')
        axes[1, 1].set_xlabel('Tours')
        axes[1, 1].set_ylabel('Nombre')
        axes[1, 1].grid(True)

    plt.tight_layout()

    # Sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(PLOTS_DIR, f"{method_name}_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Graphique sauvegardé: {plot_file}")
    return plot_file

def main():
    parser = argparse.ArgumentParser(description='Apprentissage Fédéré')
    parser.add_argument('--method', type=str, default='fedavg',
                        choices=['fedavg', 'fedprox', 'scaffold', 'compare'],
                        help='Méthode d\'agrégation')
    parser.add_argument('--iid', action='store_true', help='Distribution IID')
    parser.add_argument('--model', type=str, default='standard',
                        choices=['standard', 'lightweight', 'robust'],
                        help='Type de modèle')

    args = parser.parse_args()

    try:
        # Préparation des données
        print("Préparation des données...")
        fed_data, test_data, _ = prepare_federated_cifar10(iid=args.iid)

        # Validation des données
        if not fed_data or not test_data:
            print("Erreur: Données non préparées correctement")
            return

        if args.method == 'compare':
            print("Démarrage de la comparaison de toutes les méthodes...")
            try:
                results = compare_aggregation_methods(fed_data, test_data, args.model)
                print("\n=== RÉSULTATS DE COMPARAISON ===")
                for method, metrics in results.items():
                    acc = metrics.get('final_accuracy', 0.0)
                    print(f"{method}: Accuracy = {acc:.4f}")

                # Sauvegarde et visualisation
                save_comparison_results(results, args)
                plot_comparison_results(results, args)
            except Exception as e:
                print(f"Erreur lors de la comparaison: {e}")
                return
        else:
            # Entraînement avec une méthode spécifique
            print(f"Méthode sélectionnée: {args.method}")

            try:
                aggregator = {
                    'fedavg': FedAvgAggregator(),
                    'fedprox': FedProxAggregator(),
                    'scaffold': ScaffoldAggregator(),
                    'fedopt': FedOptAggregator(optimizer_type="adam")
                }[args.method]

                print(f"Agrégateur créé: {aggregator.name}")

            except KeyError:
                print(f"Méthode non supportée: {args.method}")
                return
            except Exception as e:
                print(f"Erreur création agrégateur: {e}")
                return

            try:
                # Créer clients et serveur
                print("Création des clients...")
                clients = [FederatedClient(i, data, aggregator)
                           for i, data in enumerate(fed_data)]

                print("Création du serveur...")
                server = FederatedServer(aggregator, args.model)

                # Validation
                if not clients:
                    print("Erreur: Aucun client créé")
                    return

                print(f"Clients créés: {len(clients)}")
                print("Démarrage de l'entraînement...")

                # Entraînement avec gestion d'erreurs
                model, metrics = server.train_federated(clients, test_data)

                if metrics and 'test_accuracy' in metrics and metrics['test_accuracy']:
                    final_acc = metrics['test_accuracy'][-1]
                    print(f"\nAccuracy finale: {final_acc:.4f}")

                    # Sauvegarde et visualisation
                    save_single_method_results(server, args, model)
                    plot_single_method_metrics(metrics, args.method, args)
                else:
                    print("\nErreur: Aucune métrique disponible")

            except Exception as e:
                print(f"Erreur lors de l'entraînement: {e}")
                import traceback
                traceback.print_exc()
                return

    except Exception as e:
        print(f"Erreur générale: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()