"""
Client MPC
"""
from FL_Agregations.client import FederatedClient
from mpc.secure_aggregation import MPCAggregator


class MPCClient(FederatedClient):
    """Client pour FL avec MPC"""

    def __init__(self, client_id, data):
        # Utiliser MPCAggregator par défaut
        super().__init__(client_id, data, MPCAggregator())
        self.is_byzantine = False

    def get_secure_update(self, global_model):
        """Prépare mise à jour pour MPC"""
        if self.is_byzantine:
            return self._generate_byzantine_update(global_model)

        return self.get_update(global_model)

    def _generate_byzantine_update(self, global_model):
        """Génère une mise à jour malveillante (pour tests)"""
        normal_update = self.get_update(global_model)
        # Amplifier l'update pour simuler une attaque
        return [update * 10 for update in normal_update]