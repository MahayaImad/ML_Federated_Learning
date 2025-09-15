"""
Modèles pour attaques MIA
"""
import tensorflow as tf
from tensorflow.keras import layers


def create_mia_model(dataset_name, input_shape, num_classes):
    """
    Crée un modèle pour les expériences MIA

    Args:
        dataset_name: 'mnist' ou 'cifar'
        input_shape: forme des données d'entrée
        num_classes: nombre de classes

    Returns:
        model: modèle Keras pour MIA
    """

    if dataset_name == 'mnist':
        return create_mnist_mia_model(input_shape, num_classes)
    elif dataset_name == 'cifar':
        return create_cifar_mia_model(input_shape, num_classes)
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}")


def create_mnist_mia_model(input_shape, num_classes):
    """Modèle CNN pour MNIST dans contexte MIA"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='MNIST_MIA_Target')

    return model


def create_cifar_mia_model(input_shape, num_classes):
    """Modèle CNN pour CIFAR-10 dans contexte MIA"""
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
    ], name='CIFAR_MIA_Target')

    return model


def create_shadow_model(dataset_name, input_shape, num_classes, model_id=0):
    """Crée un modèle shadow (légèrement différent du modèle cible)"""

    if dataset_name == 'mnist':
        # Variation du modèle MNIST
        model = tf.keras.Sequential([
            layers.Conv2D(16, (5, 5), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ], name=f'MNIST_Shadow_{model_id}')

    else:  # cifar
        # Variation du modèle CIFAR
        model = tf.keras.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation='softmax')
        ], name=f'CIFAR_Shadow_{model_id}')

    return model


def create_attack_model(input_dim, attack_type='neural'):
    """Crée un modèle pour l'attaque MIA"""

    if attack_type == 'neural':
        # Réseau de neurones pour l'attaque
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Classification binaire membre/non-membre
        ], name='Neural_MIA_Attacker')

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    else:
        # Pour autres types d'attaques, retourner None
        # (utilisation de sklearn classifiers dans attacks.py)
        return None


def create_robust_target_model(dataset_name, input_shape, num_classes, defense_level='medium'):
    """Crée un modèle cible avec défenses contre MIA"""

    if dataset_name == 'mnist':
        base_model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
        ])
    else:  # cifar
        base_model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
        ])

    # Ajout de défenses selon le niveau
    if defense_level == 'high':
        # Dropout fort + régularisation
        base_model.add(layers.Dense(128, activation='relu'))
        base_model.add(layers.Dropout(0.7))
        base_model.add(layers.Dense(64, activation='relu'))
        base_model.add(layers.Dropout(0.6))

    elif defense_level == 'medium':
        # Dropout modéré
        base_model.add(layers.Dense(256, activation='relu'))
        base_model.add(layers.Dropout(0.5))
        base_model.add(layers.Dense(128, activation='relu'))
        base_model.add(layers.Dropout(0.4))

    else:  # low
        # Dropout faible
        base_model.add(layers.Dense(512, activation='relu'))
        base_model.add(layers.Dropout(0.3))
        base_model.add(layers.Dense(256, activation='relu'))
        base_model.add(layers.Dropout(0.2))

    # Couche de sortie
    base_model.add(layers.Dense(num_classes, activation='softmax'))

    base_model._name = f'{dataset_name.upper()}_Robust_{defense_level}'

    return base_model


def add_differential_privacy_noise(model, noise_scale=0.1):
    """Ajoute du bruit pour la confidentialité différentielle (simulation)"""
    # Note: Implémentation simplifiée pour le projet académique
    # En pratique, nécessiterait TensorFlow Privacy ou implémentation complète de DP-SGD

    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            # Ajouter du bruit gaussien aux poids
            weights = layer.get_weights()
            if weights:
                noisy_weights = []
                for w in weights:
                    noise = tf.random.normal(w.shape, stddev=noise_scale)
                    noisy_w = w + noise
                    noisy_weights.append(noisy_w)
                layer.set_weights(noisy_weights)

    return model


def create_ensemble_model(dataset_name, input_shape, num_classes, num_models=3):
    """Crée un ensemble de modèles pour défense"""
    models = []

    for i in range(num_models):
        if dataset_name == 'mnist':
            # Variations architecturales pour MNIST
            if i == 0:
                model = tf.keras.Sequential([
                    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation='relu'),
                    layers.Flatten(),
                    layers.Dense(64, activation='relu'),
                    layers.Dense(num_classes, activation='softmax')
                ])
            elif i == 1:
                model = tf.keras.Sequential([
                    layers.Conv2D(16, (5, 5), activation='relu', input_shape=input_shape),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(32, (5, 5), activation='relu'),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(num_classes, activation='softmax')
                ])
            else:
                model = tf.keras.Sequential([
                    layers.Flatten(input_shape=input_shape),
                    layers.Dense(256, activation='relu'),
                    layers.Dense(128, activation='relu'),
                    layers.Dense(num_classes, activation='softmax')
                ])
        else:  # cifar
            # Variations pour CIFAR
            model = tf.keras.Sequential([
                layers.Conv2D(32 + i * 16, (3, 3), activation='relu', input_shape=input_shape),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64 + i * 16, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(256 + i * 64, activation='relu'),
                layers.Dropout(0.3 + i * 0.1),
                layers.Dense(num_classes, activation='softmax')
            ])

        model._name = f'{dataset_name.upper()}_Ensemble_{i}'
        models.append(model)

    return models