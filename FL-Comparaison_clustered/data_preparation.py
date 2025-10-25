"""
Data preparation for MNIST, CIFAR10, and MALNET datasets
"""

import numpy as np
import tensorflow as tf
from config import ALPHA


def prepare_data(dataset_name, num_clients, iid=False):
    """
    Prepare federated data for specified dataset
    
    Args:
        dataset_name: 'mnist', 'cifar10', or 'malnet'
        num_clients: Number of clients
        iid: If True, IID distribution; else non-IID
    
    Returns:
        fed_data: List of (x_train, y_train) tuples for each client
        test_data: (x_test, y_test) tuple
        client_info: List of client information dictionaries
    """
    if dataset_name == 'mnist':
        return prepare_mnist(num_clients, iid)
    elif dataset_name == 'cifar10':
        return prepare_cifar10(num_clients, iid)
    elif dataset_name == 'malnet':
        return prepare_malnet(num_clients, iid)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def prepare_mnist(num_clients, iid=False):
    """Prepare MNIST dataset"""
    print(f"Preparing MNIST data ({'IID' if iid else 'Non-IID'})")
    
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Test data (centralized)
    test_data = (x_test, y_test)
    
    # Create federated splits
    if iid:
        fed_data = create_iid_split(x_train, y_train, num_clients)
    else:
        fed_data = create_non_iid_split(x_train, y_train, num_clients, ALPHA)
    
    # Collect client info
    client_info = collect_client_info(fed_data)
    print_data_info(client_info, "MNIST")
    
    return fed_data, test_data, client_info


def prepare_cifar10(num_clients, iid=False):
    """Prepare CIFAR10 dataset"""
    print(f"Preparing CIFAR10 data ({'IID' if iid else 'Non-IID'})")
    
    # Load CIFAR10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Test data (centralized)
    test_data = (x_test, y_test)
    
    # Create federated splits
    if iid:
        fed_data = create_iid_split(x_train, y_train, num_clients)
    else:
        fed_data = create_non_iid_split(x_train, y_train, num_clients, ALPHA)
    
    # Collect client info
    client_info = collect_client_info(fed_data)
    print_data_info(client_info, "CIFAR10")
    
    return fed_data, test_data, client_info


def prepare_malnet(num_clients, iid=False):
    """
    Prepare MALNET dataset
    
    Note: This is a placeholder implementation.
    You need to provide the actual MALNET data loading logic.
    Expected format: feature vectors and labels for Android malware classification
    """
    print(f"Preparing MALNET data ({'IID' if iid else 'Non-IID'})")
    
    # TODO: Implement actual MALNET data loading
    # For now, this is a placeholder that should be replaced with actual data loading
    
    try:
        # Try to load MALNET data (you need to implement this)
        x_train, y_train, x_test, y_test = load_malnet_data()
        
        num_classes = len(np.unique(np.argmax(y_train, axis=1)))
        
        # Test data (centralized)
        test_data = (x_test, y_test)
        
        # Create federated splits
        if iid:
            fed_data = create_iid_split(x_train, y_train, num_clients)
        else:
            fed_data = create_non_iid_split(x_train, y_train, num_clients, ALPHA)
        
        # Collect client info
        client_info = collect_client_info(fed_data)
        print_data_info(client_info, "MALNET")
        
        return fed_data, test_data, client_info
        
    except Exception as e:
        print(f"Error loading MALNET data: {e}")
        print("Please implement load_malnet_data() function")
        raise NotImplementedError("MALNET data loading not implemented")


def load_malnet_data():
    """
    Load MALNET dataset
    
    TODO: Implement this function based on your MALNET data format
    
    Expected return format:
        x_train: numpy array of features (n_samples, n_features)
        y_train: numpy array of one-hot encoded labels (n_samples, n_classes)
        x_test: numpy array of test features
        y_test: numpy array of test one-hot encoded labels
    """
    raise NotImplementedError(
        "Please implement MALNET data loading.\n"
        "Expected format:\n"
        "  - x_train, x_test: Feature vectors (e.g., from static analysis)\n"
        "  - y_train, y_test: One-hot encoded labels for malware families"
    )


def create_iid_split(x_train, y_train, num_clients):
    """Create IID data split"""
    total_samples = len(x_train)
    size_per_client = total_samples // num_clients
    
    # Random shuffle
    indices = np.random.permutation(total_samples)
    
    fed_data = []
    for i in range(num_clients):
        start = i * size_per_client
        end = start + size_per_client if i < num_clients - 1 else total_samples
        
        client_indices = indices[start:end]
        x_client = x_train[client_indices]
        y_client = y_train[client_indices]
        
        fed_data.append((x_client, y_client))
    
    return fed_data


def create_non_iid_split(x_train, y_train, num_clients, alpha):
    """
    Create non-IID data split using Dirichlet distribution
    
    Args:
        x_train: Training data
        y_train: Training labels (one-hot encoded)
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (smaller = more non-IID)
    """
    num_classes = y_train.shape[1]
    labels = np.argmax(y_train, axis=1)
    
    # Group indices by class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Initialize client data indices
    client_indices = [[] for _ in range(num_clients)]
    
    # Distribute each class to clients using Dirichlet
    for c_idx in class_indices:
        np.random.shuffle(c_idx)
        
        # Sample from Dirichlet
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        
        # Split indices according to proportions
        splits = np.split(c_idx, proportions)
        
        for i, split in enumerate(splits):
            client_indices[i].extend(split)
    
    # Create federated data
    fed_data = []
    for indices in client_indices:
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        x_client = x_train[indices]
        y_client = y_train[indices]
        
        fed_data.append((x_client, y_client))
    
    return fed_data


def collect_client_info(fed_data):
    """Collect information about each client's data"""
    client_info = []
    
    for i, (x_client, y_client) in enumerate(fed_data):
        class_distribution = np.sum(y_client, axis=0)
        
        client_info.append({
            'client_id': i,
            'num_samples': len(x_client),
            'class_distribution': class_distribution
        })
    
    return client_info


def print_data_info(client_info, dataset_name):
    """Print information about data distribution"""
    print(f"\n{dataset_name} Data Distribution:")
    print("-" * 50)
    
    for info in client_info:
        top_classes = np.argsort(info['class_distribution'])[-3:][::-1]
        top_counts = info['class_distribution'][top_classes]
        
        print(f"Client {info['client_id']}: {info['num_samples']} samples")
        print(f"  Top classes: {top_classes} ({top_counts.astype(int)})")
    
    print("-" * 50)
    print()
