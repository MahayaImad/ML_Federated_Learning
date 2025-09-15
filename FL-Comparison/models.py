"""
Modèles pour comparaisons FL vs autres méthodes
"""
import tensorflow as tf
from tensorflow.keras import layers


def create_comparison_model(dataset_name, input_shape, num_classes):
    """
    Crée un modèle adapté pour les comparaisons

    Args:
        dataset_name: 'mnist' ou 'cifar'
        input_shape: forme des données d'entrée
        num_classes: nombre de classes

    Returns:
        model: modèle Keras pour comparaison
    """

    if dataset_name == 'mnist':
        return create_mnist_comparison_model(input_shape, num_classes)
    elif dataset_name == 'cifar':
        return create_cifar_comparison_model(input_shape, num_classes)
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}")


def create_mnist_comparison_model(input_shape, num_classes):
    """Modèle CNN optimisé pour MNIST dans les comparaisons"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='MNIST_Comparison_CNN')

    return model


def create_cifar_comparison_model(input_shape, num_classes):
    """Modèle CNN optimisé pour CIFAR-10 dans les comparaisons"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CIFAR_Comparison_CNN')

    return model


def create_lightweight_model(dataset_name, input_shape, num_classes):
    """Modèle léger pour clients avec ressources limitées"""
    if dataset_name == 'mnist':
        return tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ], name='MNIST_Lightweight')

    else:  # cifar
        return tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ], name='CIFAR_Lightweight')


def create_robust_model(dataset_name, input_shape, num_classes):
    """Modèle robuste avec régularisation forte"""
    if dataset_name == 'mnist':
        return tf.keras.Sequential([
            layers.Input(shape=input_shape),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            layers.Dense(num_classes, activation='softmax')
        ], name='MNIST_Robust')

    else:  # cifar
        return tf.keras.Sequential([
            # Augmentation des données
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),

            layers.Input(shape=input_shape),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.6),
            layers.Dense(num_classes, activation='softmax')
        ], name='CIFAR_Robust')