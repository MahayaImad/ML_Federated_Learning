"""
Chargement et préparation des datasets
"""

import tensorflow as tf
import numpy as np


def load_dataset(dataset_name, batch_size=64):
    """
    Charge et prépare un dataset

    Args:
        dataset_name: 'mnist' ou 'cifar'
        batch_size: taille des lots

    Returns:
        train_data: dataset d'entraînement
        test_data: dataset de test
        input_shape: forme des données d'entrée
        num_classes: nombre de classes
    """

    if dataset_name == 'mnist':
        return load_mnist(batch_size)
    elif dataset_name == 'cifar':
        return load_cifar10(batch_size)
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}")


def load_mnist(batch_size):
    """Charge et prépare MNIST"""
    print("Téléchargement de MNIST...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation MNIST standard
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = (x_train - 0.1307) / 0.3081
    x_test = (x_test - 0.1307) / 0.3081

    # Ajouter dimension canal pour CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    input_shape = x_train.shape[1:]
    num_classes = 10

    print(f"MNIST chargé: {x_train.shape[0]} échantillons d'entraînement, {x_test.shape[0]} de test")

    # Créer les datasets TensorFlow
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.shuffle(1000).batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(1000)

    return train_data, test_data, input_shape, num_classes


def load_cifar10(batch_size):
    """Charge et prépare CIFAR-10"""
    print("Téléchargement de CIFAR-10...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation CIFAR-10 standard
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train = (x_train - 0.5) / 0.5
    x_test = (x_test - 0.5) / 0.5

    # Aplatir les labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    input_shape = x_train.shape[1:]
    num_classes = 10

    print(f"CIFAR-10 chargé: {x_train.shape[0]} échantillons d'entraînement, {x_test.shape[0]} de test")

    # Créer les datasets TensorFlow
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.shuffle(1000).batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_data = test_data.batch(1000)

    return train_data, test_data, input_shape, num_classes