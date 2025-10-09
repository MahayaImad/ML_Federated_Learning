import numpy as np
import tensorflow as tf



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


def prepare_federated_cifar100(iid=True, alpha=0.5, num_clients=10):
    """
    Prépare CIFAR-100 pour l'apprentissage fédéré

    Args:
        iid: Si True, distribution IID. Si False, non-IID
        alpha: Paramètre de concentration Dirichlet pour non-IID
        num_clients: Nombre de clients

    Returns:
        fed_data: données fédérées par client
        test_data: données de test centralisées
        client_info: informations sur chaque client
    """
    print(f"Préparation des données CIFAR-100 ({'IID' if iid else 'Non-IID'})")

    # Charger CIFAR-100
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)

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

    _print_data_info(client_info, dataset="CIFAR-100")

    return fed_data, test_data, client_info

def prepare_federated_mnist(iid=True, alpha=0.5, num_clients =10):
    """
    Data preparation for federated learning experiments

    Args:
        iid: If True, distribution IID. Si False, non-IID
        alpha: index of Dirichlet for non-IID
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
    """Creates non-IID split using Dirichlet distribution """

    num_classes = y_train.shape[1]
    total_samples = len(x_train)
    samples_per_client = total_samples // num_clients

    # Group indices by class (as lists for removal)
    class_indices = {
        c: np.where(y_train.argmax(axis=1) == c)[0].tolist()
        for c in range(num_classes)
    }

    fed_data = []
    for client_id in range(num_clients):
        # Generate class proportions via Dirichlet
        proportions = np.random.dirichlet([alpha] * num_classes)
        samples_per_class = (proportions * samples_per_client).astype(int)
        samples_per_class[np.argmax(proportions)] += samples_per_client - samples_per_class.sum()

        # Collect samples from each class
        client_indices = []
        for class_id, num_samples in enumerate(samples_per_class):
            available = len(class_indices[class_id])
            if num_samples > 0 and available > 0:
                take = min(num_samples, available)
                positions = np.random.choice(available, take, replace=False)
                # Remove from pool in reverse order
                for pos in sorted(positions, reverse=True):
                    client_indices.append(class_indices[class_id].pop(pos))

        # Fill shortage from remaining samples if needed
        if len(client_indices) < samples_per_client:
            remaining = [idx for indices in class_indices.values() for idx in indices]
            shortage = samples_per_client - len(client_indices)
            if remaining and shortage > 0:
                extra = np.random.choice(len(remaining), min(shortage, len(remaining)), replace=False)
                client_indices.extend([remaining[i] for i in extra])
                # Remove used indices
                extra_set = set(client_indices[-len(extra):])
                for c in range(num_classes):
                    class_indices[c] = [i for i in class_indices[c] if i not in extra_set]

        fed_data.append((x_train[client_indices], y_train[client_indices]))

    return fed_data


def _print_data_info(client_info, dataset="CIFAR-10"):
    print(f"\n{dataset} Data Distribution")
    print("-" * 50)

    for info in client_info:
        class_dist = info['class_distribution']
        top_classes = np.argsort(class_dist)[-3:][::-1]

        print(f"Client {info['client_id']}: {info['num_samples']} samples")
        print(f"  Top classes: {top_classes} ({class_dist[top_classes].astype(int)})")

    print("-" * 50)


def number_classes(dataset="cifar10"):
    if dataset == "cifar10" or dataset == "mnist":
        return 10
    else: # CIFAR-100
        return 100


def data_shape(dataset="cifar10"):
    if dataset == "cifar10" or dataset == "cifar100":
        return (32, 32, 3)
    else: # mnist
        return (28, 28, 1)
