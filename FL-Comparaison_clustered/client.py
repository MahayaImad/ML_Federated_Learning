"""
Federated Learning Client Implementation
"""

import numpy as np
import tensorflow as tf
import time
from models import initialize_global_model


class FederatedClient:
    """Federated Learning Client"""
    
    def __init__(self, client_id, data, batch_size, dataset_name):
        """
        Initialize a federated client
        
        Args:
            client_id: Unique client identifier
            data: Tuple of (x_train, y_train)
            batch_size: Batch size for training
            dataset_name: Name of dataset
        """
        self.client_id = client_id
        self.x_train, self.y_train = data
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        
        # Initialize local model
        self.local_model = initialize_global_model(dataset_name)
        
        # Create TensorFlow dataset
        self.train_dataset = self._create_dataset()
        
        # Training statistics
        self.training_rounds = 0
        self.total_training_time = 0.0
        self.final_accuracy = 0.0
        self.final_loss = 0.0
    
    def _create_dataset(self):
        """Create TensorFlow dataset for training"""
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        )
        dataset = dataset.shuffle(buffer_size=len(self.x_train))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def train_local(self, epochs):
        """
        Train local model for specified epochs
        
        Args:
            epochs: Number of local training epochs
        """
        start_time = time.time()
        
        # Convert one-hot labels to class indices
        y_train_labels = np.argmax(self.y_train, axis=1)
        
        # Train
        history = self.local_model.fit(
            self.x_train,
            y_train_labels,
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        # Update statistics
        self.training_rounds += 1
        training_time = time.time() - start_time
        self.total_training_time += training_time
        
        if history.history:
            self.final_accuracy = history.history['accuracy'][-1]
            self.final_loss = history.history['loss'][-1]
    
    def get_client_stats(self):
        """
        Get client statistics
        
        Returns:
            Dictionary with client statistics
        """
        # Calculate class distribution
        labels = np.argmax(self.y_train, axis=1)
        num_classes = self.y_train.shape[1]
        class_counts = {}
        
        for i in range(num_classes):
            class_counts[f'class_{i}'] = int(np.sum(labels == i))
        
        stats = {
            'client_id': self.client_id,
            'data_size': len(self.x_train),
            'training_rounds': self.training_rounds,
            'total_training_time': round(self.total_training_time, 2),
            'avg_time_per_round': round(
                self.total_training_time / max(1, self.training_rounds), 2
            ),
            'final_accuracy': round(self.final_accuracy, 4),
            'final_loss': round(self.final_loss, 4)
        }
        
        # Add class distribution
        stats.update(class_counts)
        
        return stats
    
    def update_model(self, weights):
        """
        Update local model with new weights
        
        Args:
            weights: Model weights to set
        """
        self.local_model.set_weights(weights)
