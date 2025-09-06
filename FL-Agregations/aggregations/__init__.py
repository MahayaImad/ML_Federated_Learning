"""
Package d'agrégations pour l'apprentissage fédéré
"""

from .base_aggregator import BaseAggregator
from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .scaffold import ScaffoldAggregator
from .fedopt import FedOptAggregator
from .secure_aggregation import SecureAggregator

__all__ = [
    'BaseAggregator',
    'FedAvgAggregator',
    'FedProxAggregator',
    'ScaffoldAggregator',
    'FedOptAggregator',
    'SecureAggregator'
]