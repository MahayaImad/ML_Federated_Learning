"""
Implémentation des comparaisons FL vs autres méthodes
"""
import time
import tensorflow as tf
import numpy as np
from models import create_comparison_model


def compare_fl_vs_central(data_splits, args):
    """Compare apprentissage fédéré vs centralisé"""
    print("🔄 Comparaison FL vs Centralisé...")

    results = {}

    # 1. Entraînement centralisé
    print("📊 Entraînement centralisé...")
    centralized_result = train_centralized(data_splits, args)
    results['Centralized'] = centralized_result

    # 2. Entraînement fédéré
    print("🌐 Entraînement fédéré...")
    federated_result = train_federated(data_splits, args)
    results['Federated'] = federated_result

    return results


def compare_fl_vs_distributed(data_splits, args):
    """Compare apprentissage fédéré vs distribué"""
    print("🔄 Comparaison FL vs Distribué...")

    results = {}

    # 1. Entraînement fédéré
    print("🌐 Entraînement fédéré...")
    federated_result = train_federated(data_splits, args)
    results['Federated'] = federated_result

    # 2. Entraînement distribué
    print("🔗 Entraînement distribué...")
    distributed_result = train_distributed(data_splits, args)
    results['Distributed'] = distributed_result

    return results


def compare_all_methods(data_splits, args):
    """Compare toutes les méthodes"""
    print("🔄 Comparaison complète...")

    results = {}

    # 1. Centralisé
    print("📊 Entraînement centralisé...")
    results['Centralized'] = train_centralized(data_splits, args)

    # 2. Fédéré
    print("🌐 Entraînement fédéré...")
    results['Federated'] = train_federated(data_splits, args)

    # 3. Distribué
    print("🔗 Entraînement distribué...")
    results['Distributed'] = train_distributed(data_splits, args)

    # 4. Variantes
    print("🔄 Variants fédérés...")
    results['Federated_IID'] = train_federated_variant(data_splits, args, 'iid')
    results['Federated_Robust'] = train_federated_variant(data_splits, args, 'robust')

    return results


def train_centralized(data_splits, args):
    """Entraînement centralisé standard"""
    start_time = time.time()

    # Créer le modèle
    model = create_comparison_model(
        args.dataset, data_splits['input_shape'], data_splits['num_classes']
    )

    # Compiler
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraîner
    history = model.fit(
        data_splits['centralized']['train'],
        epochs=args.epochs,
        validation_data=data_splits['centralized']['test'],
        verbose=1 if args.verbose else 0
    )

    # Évaluer
    test_loss, test_acc = model.evaluate(
        data_splits['centralized']['test'], verbose=0
    )

    training_time = time.time() - start_time

    return {
        'final_accuracy': test_acc,
        'training_time': training_time,
        'history': history.history,
        'communication_cost': 0,  # Pas de communication
        'convergence_round': len(history.history['accuracy'])
    }


def train_federated(data_splits, args):
    """Entraînement fédéré simple (FedAvg)"""
    start_time = time.time()

    # Modèle global
    global_model = create_comparison_model(
        args.dataset, data_splits['input_shape'], data_splits['num_classes']
    )

    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    client_datasets = data_splits['federated']['clients']
    test_data = data_splits['federated']['test']

    accuracy_history = []
    communication_cost = 0

    # Tours de communication
    for round_num in range(args.epochs):
        round_weights = []
        round_sizes = []

        # Entraînement local pour chaque client
        for client_data in client_datasets:
            # Copier le modèle global
            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            client_model.set_weights(global_model.get_weights())

            # Entraîner localement (1 époque par tour)
            client_model.fit(client_data, epochs=1, verbose=0)

            # Collecter les poids et taille
            round_weights.append(client_model.get_weights())
            round_sizes.append(sum(1 for _ in client_data))

        # Agrégation FedAvg
        aggregated_weights = fedavg_aggregate(round_weights, round_sizes)
        global_model.set_weights(aggregated_weights)

        # Évaluer
        _, accuracy = global_model.evaluate(test_data, verbose=0)
        accuracy_history.append(accuracy)

        # Calculer coût de communication
        num_params = sum(np.prod(w.shape) for w in global_model.get_weights())
        communication_cost += num_params * len(client_datasets) * 2  # Upload + download

        if args.verbose:
            print(f"  Round {round_num + 1}: Accuracy = {accuracy:.4f}")

    training_time = time.time() - start_time

    return {
        'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0,
        'training_time': training_time,
        'history': {'accuracy': accuracy_history},
        'communication_cost': communication_cost,
        'convergence_round': len(accuracy_history)
    }


def train_distributed(data_splits, args):
    """Entraînement distribué (parallèle sans coordination)"""
    start_time = time.time()

    node_datasets = data_splits['distributed']['nodes']
    test_data = data_splits['distributed']['test']

    node_models = []
    node_accuracies = []

    # Entraîner chaque nœud indépendamment
    for i, node_data in enumerate(node_datasets):
        if args.verbose:
            print(f"  Entraînement nœud {i + 1}/{len(node_datasets)}")

        # Créer modèle pour ce nœud
        node_model = create_comparison_model(
            args.dataset, data_splits['input_shape'], data_splits['num_classes']
        )

        node_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Entraîner
        node_model.fit(node_data, epochs=args.epochs, verbose=0)

        # Évaluer
        _, accuracy = node_model.evaluate(test_data, verbose=0)
        node_models.append(node_model)
        node_accuracies.append(accuracy)

    training_time = time.time() - start_time

    # Moyenne des performances
    avg_accuracy = np.mean(node_accuracies)
    std_accuracy = np.std(node_accuracies)

    return {
        'final_accuracy': avg_accuracy,
        'accuracy_std': std_accuracy,
        'training_time': training_time,
        'history': {'accuracy': node_accuracies},
        'communication_cost': 0,  # Pas de communication entre nœuds
        'convergence_round': args.epochs
    }


def train_federated_variant(data_splits, args, variant_type):
    """Entraîne une variante du FL"""
    if variant_type == 'iid':
        # Simuler FL avec données IID (mélanger avant distribution)
        return train_federated(data_splits, args)
    elif variant_type == 'robust':
        # FL avec clients robustes (dropout simulation)
        return train_federated_with_dropout(data_splits, args)
    else:
        return train_federated(data_splits, args)


def train_federated_with_dropout(data_splits, args, dropout_rate=0.3):
    """FL avec simulation de dropout de clients"""
    start_time = time.time()

    # Même logique que train_federated mais avec dropout aléatoire
    global_model = create_comparison_model(
        args.dataset, data_splits['input_shape'], data_splits['num_classes']
    )

    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    client_datasets = data_splits['federated']['clients']
    test_data = data_splits['federated']['test']

    accuracy_history = []
    communication_cost = 0

    for round_num in range(args.epochs):
        # Sélection aléatoire des clients (simule dropout)
        available_clients = []
        for i, client_data in enumerate(client_datasets):
            if np.random.random() > dropout_rate:
                available_clients.append((i, client_data))

        if not available_clients:
            available_clients = [(0, client_datasets[0])]  # Au moins un client

        round_weights = []
        round_sizes = []

        # Entraîner les clients disponibles
        for client_id, client_data in available_clients:
            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            client_model.set_weights(global_model.get_weights())

            client_model.fit(client_data, epochs=1, verbose=0)

            round_weights.append(client_model.get_weights())
            # round_sizes.append(sum(1 for _ in client_data))
            round_sizes.append(len(list(client_data.unbatch())))

        # Agrégation
        aggregated_weights = fedavg_aggregate(round_weights, round_sizes)
        global_model.set_weights(aggregated_weights)

        # Évaluer
        _, accuracy = global_model.evaluate(test_data, verbose=0)
        accuracy_history.append(accuracy)

        # Communication (seulement clients actifs)
        num_params = sum(np.prod(w.shape) for w in global_model.get_weights())
        communication_cost += num_params * len(available_clients) * 2

    training_time = time.time() - start_time

    return {
        'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0,
        'training_time': training_time,
        'history': {'accuracy': accuracy_history},
        'communication_cost': communication_cost,
        'convergence_round': len(accuracy_history)
    }


def fedavg_aggregate(client_weights, client_sizes):
    """Agrégation FedAvg simple"""
    total_size = sum(client_sizes)

    # Calculer les poids normalisés
    weights = [size / total_size for size in client_sizes]

    # Agrégation pondérée
    aggregated = []
    for layer_idx in range(len(client_weights[0])):
        layer_sum = None
        for client_idx, client_w in enumerate(client_weights):
            layer = client_w[layer_idx] * weights[client_idx]
            if layer_sum is None:
                layer_sum = layer
            else:
                layer_sum += layer
        aggregated.append(layer_sum)

    return aggregated