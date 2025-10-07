"""
Federated learning client simulation
"""
import tensorflow as tf
import numpy as np
from models import copy_model


class FederatedClient:
    """Client for federated learning"""

    def __init__(self, client_id, data, aggregator, batch_size=64, learning_rate=0.001):
        self.client_id = client_id
        self.x_train, self.y_train = data
        self.aggregator = aggregator
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_model = None
        self.global_model_ref = None

        # Statistics tracking
        self.training_rounds = 0
        self.total_training_time = 0.0

    def update_model(self, global_model):
        """Updates local model with global model weights"""
        if self.local_model is None:
            self.local_model = copy_model(global_model, self.learning_rate)
        else:
            self.local_model.set_weights(global_model.get_weights())
        self.global_model_ref = global_model


    def train_local(self,epochs=1):
        """Local training with time tracking"""
        import time

        if self.local_model is None:
            raise ValueError("Local model not initialized")

        start_time = time.time()

        if hasattr(self.aggregator, 'train_client_with_proximal_term'):
            trained_model = self.aggregator.train_client_with_proximal_term(
                self.local_model, self.global_model_ref,
                self.x_train, self.y_train, epochs
            )
        else:
            history = self.local_model.fit(
                self.x_train, self.y_train,
                epochs=epochs,
                batch_size=self.batch_size,
                verbose=0,
                validation_split=0.1
            )
            trained_model = self.local_model

        # Update statistics
        training_time = time.time() - start_time
        self.training_rounds += 1
        self.total_training_time += training_time

        return trained_model

    def get_update(self, global_model):
        """Gets update for server"""
        return self.aggregator.prepare_client_update(
            self.client_id, self.local_model, global_model
        )

    def get_data_size(self):
        """Returns local dataset size"""
        return len(self.x_train)

    def evaluate_local(self):
        """Evaluates local model"""
        if self.local_model is None:
            return 0.0, 0.0
        loss, accuracy = self.local_model.evaluate(
            self.x_train, self.y_train, verbose=0
        )
        return loss, accuracy

    def get_class_distribution(self):
        """Returns class distribution"""
        return np.sum(self.y_train, axis=0)

    def get_client_stats(self):
        """
        Returns client statistics for CSV export
        """
        loss, accuracy = self.evaluate_local()
        class_dist = self.get_class_distribution()

        stats = {
            'client_id': self.client_id,
            'data_size': self.get_data_size(),
            'training_rounds': self.training_rounds,
            'total_training_time': round(self.total_training_time, 2),
            'avg_time_per_round': round(self.total_training_time / max(1, self.training_rounds), 2),
            'final_accuracy': round(accuracy, 4),
            'final_loss': round(loss, 4),
        }

        # Add class distribution
        for i, count in enumerate(class_dist):
            stats[f'class_{i}'] = int(count)

        return stats
