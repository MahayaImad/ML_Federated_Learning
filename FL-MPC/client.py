"""
Client MPC
"""
from federated_base import BaseFederatedClient
from mpc.secure_aggregation import MPCAggregator


class MPCClient(BaseFederatedClient):
    """Client pour FL avec MPC"""

    def __init__(self, client_id, data):
        # Utiliser MPCAggregator par défaut
        super().__init__(client_id, data)
        self.is_byzantine = False

    def get_update(self, global_model):
        """Récupère la mise à jour standard pour le serveur"""
        local_weights = self.local_model.get_weights()
        global_weights = global_model.get_weights()
        return [local_w - global_w for local_w, global_w in 
                zip(local_weights, global_weights)]
    
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