"""
Préparation des données pour les expériences d'agrégation
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split



def prepare_federated_cifar10(iid=True, alpha=0.5, num_clients = 10):
    """
    Prépare CIFAR-10 pour l'apprentissage fédéré

    Args:
        iid: Si True, distribution IID. Si False, non-IID
        alpha: Paramètre de concentration Dirichlet pour non-IID

    Returns:
        fed_data: données fédérées par client
        test_data: données de test centralisées
        client_info: informations sur chaque client
    """
    print(f"Préparation des données CIFAR-10 ({'IID' if iid else 'Non-IID'})")

    # Charger CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Données de test centralisées
    test_data = (x_test, y_test)

    if iid:
        fed_data = _create_iid_split(x_train, y_train, num_clients)
    else:
        fed_data = _create_non_iid_split(x_train, y_train, alpha, num_clients)

    # Informations sur les clients
    client_info = []
    for i, (x_client, y_client) in enumerate(fed_data):
        class_distribution = np.sum(y_client, axis=0)
        client_info.append({
            'client_id': i,
            'num_samples': len(x_client),
            'class_distribution': class_distribution
        })

    _print_data_info(client_info)

    return fed_data, test_data, client_info

def prepare_federated_mnist(iid=True, alpha=0.5, num_clients =10):
    """
    Prépare MNIST pour l'apprentissage fédéré

    Args:
        iid: Si True, distribution IID. Si False, non-IID
        alpha: Paramètre de concentration Dirichlet pour non-IID

    Returns:
        fed_data: données fédérées par client
        test_data: données de test centralisées
        client_info: informations sur chaque client
    """
    print(f"Préparation des données MNIST ({'IID' if iid else 'Non-IID'})")

    # Charger MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Ajouter dimension channel pour CNN (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Données de test centralisées
    test_data = (x_test, y_test)

    if iid:
        fed_data = _create_iid_split(x_train, y_train, num_clients)
    else:
        fed_data = _create_non_iid_split(x_train, y_train, alpha, num_clients)

    # Informations sur les clients
    client_info = []
    for i, (x_client, y_client) in enumerate(fed_data):
        class_distribution = np.sum(y_client, axis=0)
        client_info.append({
            'client_id': i,
            'num_samples': len(x_client),
            'class_distribution': class_distribution
        })

    _print_data_info(client_info, dataset="MNIST")

    return fed_data, test_data, client_info


def _create_iid_split(x_train, y_train, num_clients):
    """Crée une division IID des données"""
    total_samples = len(x_train)
    size_per_client = total_samples // num_clients

    # Échantillonner aléatoirement
    indices = np.random.choice(len(x_train), total_samples, replace=False)
    x_selected = x_train[indices]
    y_selected = y_train[indices]

    # Diviser entre clients
    fed_data = []
    for i in range(num_clients):
        start_idx = i * size_per_client
        end_idx = (i + 1) * size_per_client

        x_client = x_selected[start_idx:end_idx]
        y_client = y_selected[start_idx:end_idx]

        fed_data.append((x_client, y_client))

    return fed_data


def _create_non_iid_split(x_train, y_train, alpha, num_clients):
    """Crée une division non-IID avec distribution Dirichlet"""
    fed_data = []
    total_samples = len(x_train)
    size_per_client = total_samples // num_clients

    # Grouper par classe
    class_indices = {}
    for class_id in range(10):
        class_indices[class_id] = np.where(y_train.argmax(axis=1) == class_id)[0]

    # Distribution Dirichlet pour chaque client
    for client_id in range(num_clients):
        # Générer les proportions de classes avec Dirichlet
        proportions = np.random.dirichlet([alpha] * 10)

        # Calculer le nombre d'échantillons par classe
        samples_per_class = (proportions * size_per_client).astype(int)

        # Ajuster pour avoir exactement size_per_client échantillons
        diff = size_per_client - samples_per_class.sum()
        if diff > 0:
            samples_per_class[np.argmax(proportions)] += diff
        elif diff < 0:
            samples_per_class[np.argmax(samples_per_class)] += diff

        # Sélectionner les échantillons
        client_indices = []
        for class_id, num_samples in enumerate(samples_per_class):
            if num_samples > 0:
                available_indices = class_indices[class_id]
                selected = np.random.choice(
                    available_indices,
                    min(num_samples, len(available_indices)),
                    replace=False
                )
                client_indices.extend(selected)

        # Créer les données du client
        x_client = x_train[client_indices]
        y_client = y_train[client_indices]

        fed_data.append((x_client, y_client))

    return fed_data


def _print_data_info(client_info, dataset="CIFAR-10"):
    """Affiche les informations sur la distribution des données"""
    print(f"\n{'=' * 50}")
    print(f"DISTRIBUTION DES DONNÉES {dataset} PAR CLIENT")
    print(f"{'=' * 50}")

    for info in client_info:
        print(f"Client {info['client_id']}: {info['num_samples']} échantillons")
        class_dist = info['class_distribution']
        dominant_classes = np.argsort(class_dist)[-3:][::-1]
        print(f"  Classes dominantes: {dominant_classes} "
              f"({class_dist[dominant_classes].astype(int)})")

    print(f"{'=' * 50}")


def create_validation_split(fed_data, val_ratio=0.2):
    """
    Crée une division validation pour chaque client

    Args:
        fed_data: données fédérées
        val_ratio: ratio de validation

    Returns:
        fed_train_data: données d'entraînement
        fed_val_data: données de validation
    """
    fed_train_data = []
    fed_val_data = []

    for x_client, y_client in fed_data:
        x_train, x_val, y_train, y_val = train_test_split(
            x_client, y_client,
            test_size=val_ratio,
            stratify=y_client.argmax(axis=1),
            random_state=42
        )

        fed_train_data.append((x_train, y_train))
        fed_val_data.append((x_val, y_val))

    return fed_train_data, fed_val_data