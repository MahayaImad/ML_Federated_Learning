"""
Implémentation du partage de secret de Shamir
"""
import numpy as np
import random
from config import MPC_THRESHOLD, MPC_FIELD_SIZE


class ShamirSecretSharing:
    """Partage de secret de Shamir simplifié"""

    def __init__(self, threshold, field_size=MPC_FIELD_SIZE):
        self.threshold = threshold
        self.field_size = field_size

    def share_secret(self, secret, num_shares):
        """Partage un secret en num_shares parts"""
        # Convertir en entier pour calculs modulaires
        secret_int = int(secret * (2 ** 16)) % self.field_size

        # Générer polynôme aléatoire de degré threshold-1
        coeffs = [secret_int] + [random.randint(0, self.field_size - 1)
                                 for _ in range(self.threshold - 1)]

        # Générer les parts
        shares = []
        for i in range(1, num_shares + 1):
            y = self._evaluate_polynomial(coeffs, i)
            shares.append((i, y))

        return shares

    def reconstruct_secret(self, shares):
        """Reconstruit le secret à partir des parts"""
        if len(shares) < self.threshold:
            raise ValueError("Pas assez de parts pour reconstruction")

        # Prendre seulement les threshold premières parts
        shares = shares[:self.threshold]

        # Interpolation de Lagrange
        secret = 0
        for i, (xi, yi) in enumerate(shares):
            term = yi
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    term = (term * (-xj) * pow(xi - xj, -1, self.field_size)) % self.field_size
            secret = (secret + term) % self.field_size

        # Reconvertir en float
        return (secret / (2 ** 16)) % 1.0

    def _evaluate_polynomial(self, coeffs, x):
        """Évalue polynôme en x"""
        result = 0
        for i, coeff in enumerate(coeffs):
            result = (result + coeff * pow(x, i, self.field_size)) % self.field_size
        return result