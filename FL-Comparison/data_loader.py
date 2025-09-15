"""
Chargement des données pour comparaison FL vs autres méthodes
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def load_comparison_dataset(dataset_name, num_clients=5, iid=False, batch_size=32):
    """
    Prépare les données pour comparaison FL vs autres méthodes

    Args:
        dataset_name: 'mnist' ou 'cifar'
        num_clients: nombre de clients pour FL
        iid: distribution IID ou non-IID
        batch_size: taille des lots

    Returns:
        data_splits: dictionnaire avec toutes les configurations
    """

    if dataset_name == 'mnist':
        return load_mnist_comparison(num_clients, iid, batch_size)
    elif dataset_name == 'cifar':
        return load_cifar_comparison(num_clients, iid, batch_size)
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}")


def load_mnist_comparison(num_clients, iid, batch_size):
    """Charge MNIST pour comparaison"""
    print("📥 Chargement MNIST pour comparaison...")

    # Charger les données
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Ajouter dimension canal
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    return prepare_comparison_splits(
        x_train, y_train, x_test, y_test,
        num_clients, iid, batch_size
    )


def load_cifar_comparison(num_clients, iid, batch_size):
    """Charge CIFAR-10 pour comparaison"""
    print("📥 Chargement CIFAR-10 pour comparaison...")

    # Charger les données
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Aplatir les labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return prepare_comparison_splits(
        x_train, y_train, x_test, y_test,
        num_clients, iid, batch_size
    )


def prepare_comparison_splits(x_train, y_train, x_test, y_test, num_clients, iid, batch_size):
    """Prépare toutes les configurations de données pour comparaison"""

    data_splits = {
        'input_shape': x_train.shape[1:],
        'num_classes': len(np.unique(y_train)),
        'batch_size': batch_size
    }

    # 1. Données centralisées (toutes ensemble)
    data_splits['centralized'] = {
        'train': tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        'test': tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    }

    # 2. Données fédérées (divisées par clients)
    if iid:
        fed_splits = create_iid_split(x_train, y_train, num_clients)
    else:
        fed_splits = create_non_iid_split(x_train, y_train, num_clients)

    data_splits['federated'] = {
        'clients': [
            tf.data.Dataset.from_tensor_slices((x_client, y_client)).batch(batch_size)
            for x_client, y_client in fed_splits
        ],
        'test': tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    }

    # 3. Données distribuées (parties égales indépendantes)
    distributed_splits = create_distributed_split(x_train, y_train, num_clients)
    data_splits['distributed'] = {
        'nodes': [
            tf.data.Dataset.from_tensor_slices((x_node, y_node)).batch(batch_size)
            for x_node, y_node in distributed_splits
        ],
        'test': tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    }

    # Statistiques
    print(f"📊 Données préparées:")
    print(f"  - Centralisé: {len(x_train)} échantillons d'entraînement")
    print(f"  - Fédéré: {num_clients} clients, {'IID' if iid else 'Non-IID'}")
    print(f"  - Distribué: {num_clients} nœuds indépendants")
    print(f"  - Test: {len(x_test)} échantillons")

    return data_splits


def create_iid_split(x_train, y_train, num_clients):
    """Crée une division IID"""
    samples_per_client = len(x_train) // num_clients

    # Mélanger les données
    indices = np.random.permutation(len(x_train))
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    splits = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(x_train)

        x_client = x_shuffled[start_idx:end_idx]
        y_client = y_shuffled[start_idx:end_idx]
        splits.append((x_client, y_client))

    return splits


def create_non_iid_split(x_train, y_train, num_clients, alpha=0.5):
    """Crée une division non-IID avec Dirichlet"""
    num_classes = len(np.unique(y_train))
    samples_per_client = len(x_train) // num_clients

    # Grouper par classe
    class_indices = {}
    for class_id in range(num_classes):
        class_indices[class_id] = np.where(y_train == class_id)[0]

    splits = []
    for client_id in range(num_clients):
        # Distribution Dirichlet pour les proportions de classes
        proportions = np.random.dirichlet([alpha] * num_classes)
        samples_per_class = (proportions * samples_per_client).astype(int)

        # Ajuster pour avoir exactement samples_per_client
        diff = samples_per_client - samples_per_class.sum()
        if diff != 0:
            samples_per_class[np.argmax(proportions)] += diff

        # Sélectionner les échantillons
        client_indices = []
        for class_id, num_samples in enumerate(samples_per_class):
            if num_samples > 0:
                available = class_indices[class_id]
                selected = np.random.choice(
                    available,
                    min(num_samples, len(available)),
                    replace=False
                )
                client_indices.extend(selected)

        x_client = x_train[client_indices]
        y_client = y_train[client_indices]
        splits.append((x_client, y_client))

    return splits


def create_distributed_split(x_train, y_train, num_nodes):
    """Crée une division pour apprentissage distribué (parties égales)"""
    samples_per_node = len(x_train) // num_nodes

    splits = []
    for i in range(num_nodes):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node if i < num_nodes - 1 else len(x_train)

        x_node = x_train[start_idx:end_idx]
        y_node = y_train[start_idx:end_idx]
        splits.append((x_node, y_node))

    return splits