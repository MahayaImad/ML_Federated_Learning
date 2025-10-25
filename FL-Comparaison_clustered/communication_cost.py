"""
Communication Cost Tracking for Federated Learning
"""

from config import get_model_size


class CommunicationTracker:
    """Track communication costs in Federated Learning"""

    def __init__(self, dataset_name):
        """
        Initialize communication tracker

        Args:
            dataset_name: Name of dataset (determines model size)
        """
        self.dataset_name = dataset_name
        self.model_size_kb = get_model_size(dataset_name)

        # Communication counters
        self.client_to_server = 0  # Upload count
        self.server_to_client = 0  # Download count
        self.cluster_to_global = 0  # Cluster to global server
        self.global_to_cluster = 0  # Global to cluster server
        self.inter_cluster = 0  # Inter-cluster communications

        # History per round
        self.round_history = []

    def reset_round(self):
        """Reset counters for new round"""
        self.current_round = {
            'client_to_server': 0,
            'server_to_client': 0,
            'cluster_to_global': 0,
            'global_to_cluster': 0,
            'inter_cluster': 0
        }

    def record_client_upload(self, num_clients=1):
        """Record client uploading model to server/cluster"""
        self.current_round['client_to_server'] += num_clients
        self.client_to_server += num_clients

    def record_client_download(self, num_clients=1):
        """Record clients downloading model from server/cluster"""
        self.current_round['server_to_client'] += num_clients
        self.server_to_client += num_clients

    def record_cluster_upload(self, num_clusters=1):
        """Record cluster server uploading to global server"""
        self.current_round['cluster_to_global'] += num_clusters
        self.cluster_to_global += num_clusters

    def record_cluster_download(self, num_clusters=1):
        """Record cluster server downloading from global server"""
        self.current_round['global_to_cluster'] += num_clusters
        self.global_to_cluster += num_clusters

    def record_inter_cluster(self, num_exchanges=1):
        """Record inter-cluster communications (bidirectional)"""
        self.current_round['inter_cluster'] += num_exchanges
        self.inter_cluster += num_exchanges

    def finalize_round(self):
        """Finalize current round and store in history"""
        round_cost = self.calculate_round_cost(self.current_round)
        self.current_round['total_cost_kb'] = round_cost
        self.round_history.append(self.current_round.copy())
        return round_cost

    def calculate_round_cost(self, round_data):
        """
        Calculate total communication cost for a round in KB

        Args:
            round_data: Dictionary with communication counts

        Returns:
            Total cost in KB
        """
        cost = 0

        # Client <-> Server/Cluster
        cost += round_data['client_to_server'] * self.model_size_kb  # Upload
        cost += round_data['server_to_client'] * self.model_size_kb  # Download

        # Cluster <-> Global
        cost += round_data['cluster_to_global'] * self.model_size_kb  # Upload
        cost += round_data['global_to_cluster'] * self.model_size_kb  # Download

        # Inter-cluster (bidirectional)
        cost += round_data['inter_cluster'] * 2 * self.model_size_kb

        return cost

    def get_total_cost(self):
        """Get total communication cost across all rounds in KB"""
        return sum(r['total_cost_kb'] for r in self.round_history)

    def get_total_cost_mb(self):
        """Get total communication cost in MB"""
        return self.get_total_cost() / 1024

    def get_total_cost_gb(self):
        """Get total communication cost in GB"""
        return self.get_total_cost_mb() / 1024

    def get_average_cost_per_round(self):
        """Get average communication cost per round in KB"""
        if not self.round_history:
            return 0
        return self.get_total_cost() / len(self.round_history)

    def get_communication_breakdown(self):
        """Get breakdown of communication types"""
        total_cost = self.get_total_cost()

        if total_cost == 0:
            return {
                'client_upload_pct': 0,
                'client_download_pct': 0,
                'cluster_upload_pct': 0,
                'cluster_download_pct': 0,
                'inter_cluster_pct': 0
            }

        # Calculate costs for each type
        client_upload = self.client_to_server * self.model_size_kb
        client_download = self.server_to_client * self.model_size_kb
        cluster_upload = self.cluster_to_global * self.model_size_kb
        cluster_download = self.global_to_cluster * self.model_size_kb
        inter_cluster = self.inter_cluster * 2 * self.model_size_kb

        return {
            'client_upload_pct': (client_upload / total_cost) * 100,
            'client_download_pct': (client_download / total_cost) * 100,
            'cluster_upload_pct': (cluster_upload / total_cost) * 100,
            'cluster_download_pct': (cluster_download / total_cost) * 100,
            'inter_cluster_pct': (inter_cluster / total_cost) * 100
        }

    def get_summary(self):
        """Get summary statistics"""
        return {
            'total_rounds': len(self.round_history),
            'total_cost_kb': self.get_total_cost(),
            'total_cost_mb': self.get_total_cost_mb(),
            'total_cost_gb': self.get_total_cost_gb(),
            'avg_cost_per_round_kb': self.get_average_cost_per_round(),
            'model_size_kb': self.model_size_kb,
            'total_client_uploads': self.client_to_server,
            'total_client_downloads': self.server_to_client,
            'total_cluster_uploads': self.cluster_to_global,
            'total_cluster_downloads': self.global_to_cluster,
            'total_inter_cluster': self.inter_cluster,
            'breakdown': self.get_communication_breakdown()
        }

    def print_summary(self):
        """Print communication cost summary"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("COMMUNICATION COST SUMMARY")
        print("=" * 60)
        print(f"Total Rounds: {summary['total_rounds']}")
        print(f"Model Size: {summary['model_size_kb']} KB")
        print(f"\nTotal Communication Cost:")
        print(f"  {summary['total_cost_kb']:.2f} KB")
        print(f"  {summary['total_cost_mb']:.2f} MB")
        print(f"  {summary['total_cost_gb']:.4f} GB")
        print(f"\nAverage per Round: {summary['avg_cost_per_round_kb']:.2f} KB")
        print(f"\nCommunication Counts:")
        print(f"  Client Uploads: {summary['total_client_uploads']}")
        print(f"  Client Downloads: {summary['total_client_downloads']}")
        print(f"  Cluster Uploads: {summary['total_cluster_uploads']}")
        print(f"  Cluster Downloads: {summary['total_cluster_downloads']}")
        print(f"  Inter-cluster: {summary['total_inter_cluster']}")

        breakdown = summary['breakdown']
        print(f"\nCommunication Breakdown:")
        print(f"  Client Upload: {breakdown['client_upload_pct']:.1f}%")
        print(f"  Client Download: {breakdown['client_download_pct']:.1f}%")
        print(f"  Cluster Upload: {breakdown['cluster_upload_pct']:.1f}%")
        print(f"  Cluster Download: {breakdown['cluster_download_pct']:.1f}%")
        print(f"  Inter-cluster: {breakdown['inter_cluster_pct']:.1f}%")
        print("=" * 60)