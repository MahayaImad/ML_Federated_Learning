"""
Définitions des modèles CNN et MLP
"""

import tensorflow as tf
from tensorflow.keras import layers


def create_model(model_type, dataset, input_shape, num_classes):
    """
    Crée un modèle selon le type et dataset spécifiés

    Args:
        model_type: 'cnn' ou 'mlp'
        dataset: 'mnist' ou 'cifar'
        input_shape: forme des données d'entrée
        num_classes: nombre de classes

    Returns:
        model: modèle Keras compilé
    """

    if model_type == 'cnn':
        if dataset == 'mnist':
            return create_cnn_mnist(input_shape, num_classes)
        elif dataset == 'cifar':
            return create_cnn_cifar(input_shape, num_classes)
    elif model_type == 'mlp':
        return create_mlp(input_shape, num_classes)

    raise ValueError(f"Combinaison non supportée: {model_type} + {dataset}")


def create_mlp(input_shape, num_classes, dim_hidden=64):
    """Multi-Layer Perceptron simple"""
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(dim_hidden, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='MLP')

    return model


def create_cnn_mnist(input_shape, num_classes):
    """CNN optimisé pour MNIST (images N&B 28x28)"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_MNIST')

    return model


def create_cnn_cifar(input_shape, num_classes):
    """CNN optimisé pour CIFAR-10 (images couleur 32x32)"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='CNN_CIFAR')

    return model