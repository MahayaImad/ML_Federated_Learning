"""
Définitions des modèles pour l'apprentissage fédéré
"""
import tensorflow as tf
from config import WIDTH, HEIGHT, CHANNELS, NUM_CLASSES, LEARNING_RATE, DATASET_CONFIG


def create_cifar10_cnn(model_type="standard"):
    """
    Crée un modèle CNN pour CIFAR-10

    Args:
        model_type: Type de modèle ("standard", "lightweight", "robust")

    Returns:
        model: Modèle compilé
    """
    if model_type == "standard":
        return _create_standard_cnn()
    elif model_type == "lightweight":
        return _create_lightweight_cnn()
    elif model_type == "robust":
        return _create_robust_cnn()
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")


def _create_standard_cnn():
    """Modèle CNN standard"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS)),

        # Premier bloc convolutionnel
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.SpatialDropout2D(0.25),

        # Deuxième bloc convolutionnel
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.SpatialDropout2D(0.25),

        # Troisième bloc convolutionnel
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.SpatialDropout2D(0.25),

        # Couches denses
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def _create_lightweight_cnn():
    """Modèle CNN léger pour clients avec ressources limitées"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS)),

        tf.keras.layers.Conv2D(16, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def _create_robust_cnn():
    """Modèle CNN robuste avec régularisation forte"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WIDTH, HEIGHT, CHANNELS)),

        # Augmentation des données intégrée
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),

        # Architecture robuste
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.SpatialDropout2D(0.3),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.SpatialDropout2D(0.4),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.SpatialDropout2D(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_mnist_cnn(model_type="lightweight"):
    """
    Crée un modèle CNN pour MNIST (28x28x1)
    """
    if model_type == "standard":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(DATASET_CONFIG['mnist']['input_shape']),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    elif model_type == "lightweight":
        model = tf.keras.Sequential([
            tf.keras.layers.Input(DATASET_CONFIG['mnist']['input_shape']),
            tf.keras.layers.Conv2D(16, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


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