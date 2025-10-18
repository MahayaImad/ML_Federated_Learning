import time
import numpy as np


class FedAvgAggregator:
    """Standard FedAvg aggregator with support for transfer learning"""

    def __init__(self):
        self.name = "FedAvg"
        self.round_number = 0
        self.history = {
            'communication_costs': [],
            'aggregation_times': []
        }

    def aggregate(self, client_updates, client_weights, global_model, trainable_only=False):
        """
        Aggregate client updates using FedAvg

        Args:
            client_updates: List of client weight updates
            client_weights: Data sizes for weighted averaging
            global_model: Current global model
            trainable_only: If True, only aggregate trainable weights

        Returns:
            new_global_weights: Updated global model weights
        """
        start_time = time.time()

        if trainable_only:
            # Aggregate only trainable weights
            aggregated_update = weighted_average(client_updates, client_weights)

            # Get current global weights
            from models import get_trainable_weights, set_trainable_weights
            _, trainable_indices = get_trainable_weights(global_model)

            # Apply aggregated trainable weights
            set_trainable_weights(global_model, aggregated_update, trainable_indices)
            new_global_weights = global_model.get_weights()
        else:
            # Standard aggregation (all weights)
            aggregated_update = weighted_average(client_updates, client_weights)

            # Apply update to global model
            global_weights = global_model.get_weights()
            new_global_weights = [
                global_w + update for global_w, update in zip(global_weights, aggregated_update)
            ]

        # Record metrics
        comm_cost = self.get_communication_cost(client_updates)
        agg_time = time.time() - start_time

        self.history['communication_costs'].append(comm_cost)
        self.history['aggregation_times'].append(agg_time)

        return new_global_weights

    def prepare_client_update(self, client_id, local_model, global_model, trainable_only=False):
        """
        Prepare client update (weight delta or trainable weights)

        Args:
            client_id: Client ID
            local_model: Trained local model
            global_model: Global model
            trainable_only: If True, return only trainable weights

        Returns:
            client_update: Weight delta or trainable weights only
        """
        if trainable_only:
            from models import get_trainable_weights
            trainable_weights, _ = get_trainable_weights(local_model)
            return trainable_weights
        else:
            local_weights = local_model.get_weights()
            global_weights = global_model.get_weights()

            # Calculate weight delta
            client_update = subtract_weights(local_weights, global_weights)

            return client_update

    def aggregate_trainable_only(self, client_trainable_weights, client_data_sizes):
        """
        Aggregate only trainable weights from clients

        Args:
            client_trainable_weights: List of trainable weight lists from clients
            client_data_sizes: Data sizes for weighted averaging

        Returns:
            aggregated_weights: Aggregated trainable weights
        """
        start_time = time.time()

        # Weighted average
        aggregated_weights = weighted_average(client_trainable_weights, client_data_sizes)

        # Record metrics
        comm_cost = self.get_communication_cost(client_trainable_weights)
        agg_time = time.time() - start_time

        self.history['communication_costs'].append(comm_cost)
        self.history['aggregation_times'].append(agg_time)

        return aggregated_weights

    def update_round(self):
        """Update round number"""
        self.round_number += 1

    def reset(self):
        """Reset aggregator state"""
        self.round_number = 0
        self.history = {
            'communication_costs': [],
            'aggregation_times': []
        }

    def get_communication_cost(self, client_updates):
        """Calculate communication cost with robust format handling"""
        total_params = 0

        try:
            for update in client_updates:
                if isinstance(update, dict):
                    # Dictionary format (e.g., SCAFFOLD)
                    for key, value in update.items():
                        if isinstance(value, list):
                            for item in value:
                                if hasattr(item, 'shape') and hasattr(item, 'size'):
                                    total_params += item.size
                elif isinstance(update, list):
                    # Standard format (list of numpy arrays)
                    for layer_update in update:
                        if hasattr(layer_update, 'shape') and hasattr(layer_update, 'size'):
                            total_params += layer_update.size
                elif hasattr(update, 'shape') and hasattr(update, 'size'):
                    # Single array
                    total_params += update.size

        except Exception as e:
            print(f"Error calculating communication cost: {e}")
            return 0

        return total_params

    def get_stats(self):
        """Return aggregator statistics"""
        return {
            'name': self.name,
            'rounds_completed': self.round_number,
            'total_communication_cost': sum(self.history['communication_costs']),
            'avg_aggregation_time': np.mean(self.history['aggregation_times']) if self.history[
                'aggregation_times'] else 0
        }


# Helper functions

def weighted_average(updates, weights):
    """
    Calculate weighted average of updates

    Args:
        updates: List of updates
        weights: Weight for each update

    Returns:
        averaged_update: Averaged update
    """
    if not updates:
        return None

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Calculate weighted average
    averaged_update = []
    for layer_idx in range(len(updates[0])):
        layer_updates = [update[layer_idx] for update in updates]

        weighted_sum = np.zeros_like(layer_updates[0])
        for update, weight in zip(layer_updates, weights):
            weighted_sum += update * weight

        averaged_update.append(weighted_sum)

    return averaged_update


def subtract_weights(weights1, weights2):
    """Subtract weights2 from weights1"""
    return [w1 - w2 for w1, w2 in zip(weights1, weights2)]


def add_weights(weights1, weights2):
    """Add weights1 and weights2"""
    return [w1 + w2 for w1, w2 in zip(weights1, weights2)]


def scale_weights(weights, factor):
    """Multiply weights by a factor"""
    return [w * factor for w in weights]