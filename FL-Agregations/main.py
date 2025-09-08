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
from aggregations import FedAvgAggregator, FedProxAggregator, ScaffoldAggregator
from client import FederatedClient
from server import FederatedServer
from experiments.run_comparison import compare_aggregation_methods

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

    # Préparation des données
    fed_data, test_data, _ = prepare_federated_cifar10(iid=args.iid)

    if args.method == 'compare':
        results = compare_aggregation_methods(fed_data, test_data, args.model)
        print("\n=== RÉSULTATS DE COMPARAISON ===")
        for method, metrics in results.items():
            print(f"{method}: Précision = {metrics['final_accuracy']:.4f}")
    else:
        # Entraînement avec une méthode spécifique
        aggregator = {
            'fedavg': FedAvgAggregator(),
            'fedprox': FedProxAggregator(),
            'scaffold': ScaffoldAggregator()
        }[args.method]

        # Créer clients et serveur
        clients = [FederatedClient(i, data, aggregator)
                   for i, data in enumerate(fed_data)]
        server = FederatedServer(aggregator, args.model)

        # Entraînement
        model, metrics = server.train_federated(clients, test_data)

        print(f"\nPrécision finale: {metrics['test_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()