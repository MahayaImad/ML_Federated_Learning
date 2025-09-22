"""
Chargement des données pour attaques MIA
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def load_mia_dataset(dataset_name, num_clients=5, target_samples=1000, batch_size=32):
    """
    Prépare les données pour attaques MIA

    Args:
        dataset_name: 'mnist' ou 'cifar'
        num_clients: nombre de clients pour FL
        target_samples: nombre d'échantillons cibles pour MIA
        batch_size: taille des lots

    Returns:
        mia_data: dictionnaire avec données organisées pour MIA
    """

    if dataset_name == 'mnist':
        return load_mnist_mia(num_clients, target_samples, batch_size)
    elif dataset_name == 'cifar':
        return load_cifar_mia(num_clients, target_samples, batch_size)
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}")


def load_mnist_mia(num_clients, target_samples, batch_size):
    """Charge MNIST pour attaques MIA"""
    print("🎯 Préparation MNIST pour attaques MIA...")

    # Charger les données
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Ajouter dimension canal
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return prepare_mia_splits(
        x_train, y_train, x_test, y_test,
        num_clients, target_samples, batch_size
    )


def load_cifar_mia(num_clients, target_samples, batch_size):
    """Charge CIFAR-10 pour attaques MIA"""
    print("🎯 Préparation CIFAR-10 pour attaques MIA...")

    # Charger les données
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Aplatir les labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return prepare_mia_splits(
        x_train, y_train, x_test, y_test,
        num_clients, target_samples, batch_size
    )


def prepare_mia_splits(x_train, y_train, x_test, y_test, num_clients, target_samples, batch_size):
    """Prépare les divisions de données pour MIA"""

    mia_data = {
        'input_shape': x_train.shape[1:],
        'num_classes': len(np.unique(y_train)),
        'batch_size': batch_size,
        'target_samples': target_samples
    }

    # 1. Sélectionner les échantillons cibles pour MIA
    target_indices = np.random.choice(len(x_train), target_samples, replace=False)
    target_x = x_train[target_indices]
    target_y = y_train[target_indices]

    # Créer masque pour les données membres/non-membres
    member_mask = np.zeros(len(x_train), dtype=bool)
    member_mask[target_indices] = True

    mia_data['target_data'] = {
        'x': target_x,
        'y': target_y,
        'indices': target_indices,
        'member_mask': member_mask
    }

    # 2. Données d'entraînement pour modèle cible (excluant échantillons de test MIA)
    remaining_indices = np.setdiff1d(np.arange(len(x_train)), target_indices)
    train_x = x_train[remaining_indices]
    train_y = y_train[remaining_indices]

    # 3. Division fédérée des données d'entraînement
    federated_splits = create_federated_split_mia(train_x, train_y, num_clients)

    mia_data['federated_data'] = {
        'clients': [
            tf.data.Dataset.from_tensor_slices((x_client, y_client)).batch(batch_size)
            for x_client, y_client in federated_splits
        ],
        'raw_clients': federated_splits  # Pour shadow models
    }

    # 4. Données de test standard
    mia_data['test_data'] = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # 5. Données pour shadow models (sous-ensembles des données disponibles)
    shadow_data = prepare_shadow_data(x_train, y_train, target_samples)
    mia_data['shadow_data'] = shadow_data

    # 6. Échantillons non-membres pour attaques (partie des données de test)
    non_member_indices = np.random.choice(len(x_test), target_samples, replace=False)
    non_member_x = x_test[non_member_indices]
    non_member_y = y_test[non_member_indices]

    mia_data['non_member_data'] = {
        'x': non_member_x,
        'y': non_member_y,
        'indices': non_member_indices
    }

    print(f"🎯 Données MIA préparées:")
    print(f"  - Échantillons cibles (membres): {target_samples}")
    print(f"  - Échantillons non-membres: {target_samples}")
    print(f"  - Clients fédérés: {num_clients}")
    print(f"  - Données d'entraînement restantes: {len(train_x)}")

    return mia_data


def create_federated_split_mia(x_train, y_train, num_clients, alpha=0.5):
    """Crée une division fédérée pour MIA (non-IID par défaut)"""
    num_classes = len(np.unique(y_train))

    # Vérifier que samples_per_client n'est pas trop grand
    max_possible = len(x_train) // max(1, num_clients)
    samples_per_client = max_possible

    if samples_per_client == 0:
        print(f"⚠️ Pas assez de données pour {num_clients} clients")
        samples_per_client = 1

    print(f"📊 Division fédérée: {samples_per_client} échantillons par client")

    # Grouper par classe
    class_indices = {}
    for class_id in range(num_classes):
        class_indices[class_id] = np.where(y_train == class_id)[0].copy()

    splits = []
    for client_id in range(num_clients):
        # Distribution Dirichlet pour non-IID
        proportions = np.random.dirichlet([alpha] * num_classes)
        samples_per_class = (proportions * samples_per_client).astype(int)

        # Ajuster pour avoir exactement samples_per_client
        diff = samples_per_client - samples_per_class.sum()
        if diff != 0:
            # Trouver une classe qui a encore des échantillons
            available_classes = [i for i in range(num_classes) if len(class_indices[i]) > 0]
            if available_classes:
                target_class = available_classes[np.argmax([proportions[i] for i in available_classes])]
                samples_per_class[target_class] += diff
            else:
                print(f"⚠️ Plus d'échantillons disponibles pour client {client_id}")
                break

        # Sélectionner les échantillons
        client_indices = []
        for class_id, num_samples in enumerate(samples_per_class):
            if num_samples > 0 and len(class_indices[class_id]) > 0:
                available = class_indices[class_id]
                take_samples = min(num_samples, len(available))

                if take_samples > 0:
                    selected = np.random.choice(
                        available,
                        take_samples,
                        replace=False
                    )
                    client_indices.extend(selected)
                    # Retirer les indices utilisés
                    class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected)

                    # Vérifier s'il reste des indices
                    if len(class_indices[class_id]) == 0:
                        print(f"⚠️ Classe {class_id} épuisée après client {client_id}")

        # S'assurer qu'il y a au moins quelques échantillons
        if len(client_indices) == 0:
            print(f"⚠️ Client {client_id} sans données, redistribution...")
            # Prendre quelques échantillons au hasard des classes disponibles
            available_classes = [cid for cid, indices in class_indices.items() if len(indices) > 0]
            if available_classes:
                class_id = np.random.choice(available_classes)
                take_samples = min(10, len(class_indices[class_id]))
                if take_samples > 0:
                    selected = np.random.choice(class_indices[class_id], take_samples, replace=False)
                    client_indices.extend(selected)
                    class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected)

        if len(client_indices) > 0:
            x_client = x_train[client_indices]
            y_client = y_train[client_indices]
            splits.append((x_client, y_client))
            print(f"  Client {client_id}: {len(client_indices)} échantillons")
        else:
            print(f"⚠️ Client {client_id} ignoré (pas de données)")

    # S'assurer qu'il y a au moins un split
    if len(splits) == 0:
        print("⚠️ Aucun split créé, création d'un split minimal...")
        # Prendre toutes les données pour un seul client
        splits.append((x_train, y_train))

    print(f"✅ {len(splits)} clients créés avec {[len(x) for x, y in splits]} échantillons")
    return splits

def prepare_shadow_data(x_all, y_all, target_samples, num_shadow_datasets=5):
    """Prépare les données pour les modèles shadow"""
    shadow_datasets = []

    total_needed = target_samples * num_shadow_datasets * 2  # x2 pour membre/non-membre

    if total_needed > len(x_all):
        print(f"⚠️ Pas assez de données pour {num_shadow_datasets} shadow models")
        num_shadow_datasets = min(3, len(x_all) // (target_samples * 2))

    for i in range(num_shadow_datasets):
        # Sélectionner un sous-ensemble aléatoire
        start_idx = i * (target_samples * 2)
        end_idx = start_idx + (target_samples * 2)

        if end_idx > len(x_all):
            # Réutiliser des données avec permutation
            indices = np.random.choice(len(x_all), target_samples * 2, replace=False)
        else:
            indices = np.arange(start_idx, end_idx)

        shadow_x = x_all[indices]
        shadow_y = y_all[indices]

        # Diviser en membres/non-membres pour ce shadow model
        mid_point = len(shadow_x) // 2

        shadow_dataset = {
            'train_x': shadow_x[:mid_point],
            'train_y': shadow_y[:mid_point],
            'member_x': shadow_x[:mid_point],
            'member_y': shadow_y[:mid_point],
            'non_member_x': shadow_x[mid_point:],
            'non_member_y': shadow_y[mid_point:]
        }

        shadow_datasets.append(shadow_dataset)

    return shadow_datasets


def create_attack_dataset(member_data, non_member_data, predictions_member, predictions_non_member):
    """Crée le dataset pour entraîner l'attaquant MIA"""

    # Combiner les prédictions
    all_predictions = np.vstack([predictions_member, predictions_non_member])

    # Labels: 1 pour membre, 0 pour non-membre
    attack_labels = np.hstack([
        np.ones(len(predictions_member)),
        np.zeros(len(predictions_non_member))
    ])

    # Mélanger les données
    indices = np.random.permutation(len(all_predictions))
    all_predictions = all_predictions[indices]
    attack_labels = attack_labels[indices]

    return all_predictions, attack_labels


def prepare_threshold_attack_data(target_model, member_data, non_member_data):
    """Prépare les données pour l'attaque threshold"""

    # Prédictions sur les données membres
    member_predictions = target_model.predict(member_data['x'], verbose=0)
    member_confidences = np.max(member_predictions, axis=1)

    # Prédictions sur les données non-membres
    non_member_predictions = target_model.predict(non_member_data['x'], verbose=0)
    non_member_confidences = np.max(non_member_predictions, axis=1)

    return {
        'member_confidences': member_confidences,
        'non_member_confidences': non_member_confidences,
        'member_predictions': member_predictions,
        'non_member_predictions': non_member_predictions
    }