"""
Modèles pour comparaisons FL vs autres méthodes
"""
import tensorflow as tf
from tensorflow.keras import layers
from data_preparation import number_classes, data_shape


def create_model(dataset_name, input_shape):
    """
    Crée un modèle adapté pour les comparaisons

    Args:
        dataset_name: 'mnist' ou 'cifar'
        input_shape: forme des données d'entrée
        num_classes: nombre de classes

    Returns:
        model: modèle Keras pour comparaison
    """
    num_classes = number_classes()
    if dataset_name == 'mnist':
        return create_mnist_comparison_model(input_shape, num_classes)
    elif dataset_name == 'cifar10':
        return create_cifar_comparison_model(input_shape, num_classes)
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}")


def create_mnist_comparison_model(input_shape, num_classes=10):
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


def copy_model(model, learning_rate=0.001):

    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Créer un nouveau modèle avec la même architecture
        model_copy = tf.keras.models.clone_model(model)

        # Compiler le modèle
        model_copy.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Copier les poids en s'assurant qu'ils sont sur le bon device
        weights = model.get_weights()
        model_copy.set_weights(weights)

        return model_copy


def initialize_global_model(dataset_name, learning_rate=0.001):

    input_shape = data_shape()

    # Create model
    model = create_model(dataset_name, input_shape)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def initialize_edge_models(edge_servers, dataset_name, global_model, learning_rate=0.001):

    input_shape = data_shape()

    for edge in edge_servers:
        if edge.local_model is None:
            edge.local_model = create_model(dataset_name, input_shape)
            edge.local_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        edge.local_model.set_weights(global_model.get_weights())
