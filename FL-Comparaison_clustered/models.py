"""
Neural network models for different datasets
"""

import tensorflow as tf
from config import get_dataset_config


def initialize_global_model(dataset_name):
    """
    Initialize global model based on dataset
    
    Args:
        dataset_name: 'mnist', 'cifar10', or 'malnet'
    
    Returns:
        Compiled Keras model
    """
    config = get_dataset_config(dataset_name)
    
    if dataset_name in ['mnist', 'cifar10']:
        model = create_cnn_model(
            config['input_shape'],
            config['num_classes']
        )
    elif dataset_name == 'malnet':
        # For MALNET, we need to determine input shape from data
        # This is a placeholder - adjust based on your feature extraction
        model = create_dnn_model(
            config['num_classes'],
            input_dim=1024  # Adjust based on your features
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return model


def create_cnn_model(input_shape, num_classes):
    """
    Create CNN model for image classification (MNIST, CIFAR10)
    
    Args:
        input_shape: Input shape tuple (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                               input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_dnn_model(num_classes, input_dim=1024):
    """
    Create DNN model for MALNET (feature-based classification)
    
    Args:
        num_classes: Number of malware families
        input_dim: Dimension of input features
    
    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
