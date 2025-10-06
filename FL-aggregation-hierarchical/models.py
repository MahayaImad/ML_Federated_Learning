"""
Modèles pour comparaisons FL vs autres méthodes
"""
import tensorflow as tf
from tensorflow.keras import layers
from config import LEARNING_RATE


def create_model(dataset_name, input_shape, num_classes):
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
    elif dataset_name == 'cifar10':
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


def get_model_weights(model):
    """Récupère les poids d'un modèle"""
    return model.get_weights()


def set_model_weights(model, weights):
    """Définit les poids d'un modèle"""
    model.set_weights(weights)


def copy_model(model):
    """Copie sécurisée d'un modèle pour éviter les erreurs GPU/CPU"""
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Créer un nouveau modèle avec la même architecture
        model_copy = tf.keras.models.clone_model(model)

        # Compiler le modèle
        model_copy.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Copier les poids en s'assurant qu'ils sont sur le bon device
        weights = model.get_weights()
        model_copy.set_weights(weights)

        # Initialiser le modèle avec un batch factice

        # dummy_input = tf.zeros((1, WIDTH, HEIGHT, CHANNELS))
        # dummy_output = tf.zeros((1, NUM_CLASSES))

        # with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        #     model_copy(dummy_input, training=False)

        return model_copy