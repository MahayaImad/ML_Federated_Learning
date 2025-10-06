"""
Utilitaires pour comparaisons FL vs autres méthodes
"""
import os
import datetime
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import entropy


def setup_gpu(gpu_id):
    """Configure GPU usage"""
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpu_id == -1:
        print("Using CPU only")
        tf.config.set_visible_devices([], 'GPU')
    elif gpus and gpu_id < len(gpus):
        try:
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"Using GPU: {gpus[gpu_id].name}")
        except RuntimeError as e:
            print(f"GPU error: {e}")
            print("Falling back to CPU")
    else:
        print(f"GPU {gpu_id} not found, using CPU")


def save_results(results, args):
    """Sauvegarde les résultats de comparaison"""
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    base_name = f"{args.dataset}_{args.hierarchy_type}_{args.clients}cl_{args.local_epochs}ep_{args.rounds}r"

    # 1. Sauvegarder résultats détaillés
    log_path = os.path.join('results', f"{timestamp}_{base_name}.txt")
    _save_detailed_results(results, args, log_path)

    # 2. Créer visualisations
    plot_path = os.path.join('visualizations', f"{base_name}_{timestamp}.png")
    _plot_comparison_results(results, args, plot_path)

    print(f"\nResults saved to: {log_path}")
    print(f"\nVisualization saved to: {plot_path}")

def _save_detailed_results(results, args, log_path):
    """Sauvegarde les résultats détaillés au format texte"""
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Hierarchical Aggregation Results: {getattr(args, 'dataset', 'unknown').upper()}\n")
            f.write(f"Method: {results.get('method', 'unknown')}\n")
            f.write(f"Configuration: {getattr(args, 'clients', 'N/A')} clients\n")
            f.write(f"Data Distribution: {getattr(args, 'iid', 'N/A')} iid\n")
            f.write(f"Training: {getattr(args, 'epochs', 'N/A')} epochs\n")
            f.write(f"Training: {getattr(args, 'rounds', 'N/A')} rounds\n")
            f.write(f"Model: {getattr(args, 'lr', 'N/A')} Learning rate\n")
            f.write(f"Model: {getattr(args, 'batch_size', 'N/A')} batch_size\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")

            # Extraire les métriques depuis les listes
            accuracy_history = results.get('accuracy_history', [])
            communication_costs = results.get('communication_costs', [])
            round_times = results.get('round_times', [])

            # Calculer les métriques finales
            final_accuracy = accuracy_history[-1] if accuracy_history else 0.0
            total_training_time = sum(round_times) if round_times else 0.0
            total_communication_cost = sum(communication_costs) if communication_costs else 0
            convergence_rounds = len(accuracy_history)

            f.write("RÉSULTATS FINAUX:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Final Accuracy: {final_accuracy:.6f}\n")
            f.write(f"Total Training Time: {total_training_time:.2f}s\n")
            avg_time_per_round = (total_training_time / convergence_rounds) if convergence_rounds > 0 else 0.0
            f.write(f"Average Time per Round: {avg_time_per_round:.2f}s\n")
            f.write(f"Total Communication Cost: {total_communication_cost}\n")
            f.write(f"Convergence Rounds: {convergence_rounds}\n\n")

            f.write("HISTORIQUE DÉTAILLÉ:\n")
            f.write("-" * 30 + "\n")
            for i in range(len(accuracy_history)):
                round_num = i + 1
                acc = accuracy_history[i]
                time_val = round_times[i] if i < len(round_times) else 0
                comm = communication_costs[i] if i < len(communication_costs) else 0
                f.write(f"Round {round_num:2d}: Accuracy={acc:.6f}, Time={time_val:.2f}s, Comm={comm}\n")

    except Exception as e:
        print(f"❌ Erreur sauvegarde résultats: {e}")
        import traceback
        traceback.print_exc()

def _plot_comparison_results(results, args, plot_path):
    """Crée les visualisations de comparaison"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend non-interactif

        if not results:
            print("Aucun résultat à visualiser")
            return

        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # Adapter selon la structure de vos résultats
        if 'method' in results:
            # Structure FL-aggregation-hierarchical (un seul résultat)
            method_name = results.get('method', 'Unknown')
            accuracy_history = results.get('accuracy_history', [])
            round_times = results.get('round_times', [])
            communication_costs = results.get('communication_costs', [])

            # Créer le graphique pour une seule méthode
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Accuracy au fil du temps
            if accuracy_history:
                axes[0, 0].plot(range(1, len(accuracy_history) + 1), accuracy_history, 'b-o')
                axes[0, 0].set_title(f'Accuracy Evolution - {method_name}')
                axes[0, 0].set_xlabel('Rounds')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].grid(True)

            # 2. Temps par round
            if round_times:
                axes[0, 1].bar(range(1, len(round_times) + 1), round_times, color='orange')
                axes[0, 1].set_title(f'Training Time per Round - {method_name}')
                axes[0, 1].set_xlabel('Rounds')
                axes[0, 1].set_ylabel('Time (s)')
                axes[0, 1].grid(True)

            # 3. Communication costs
            if communication_costs:
                axes[1, 0].bar(range(1, len(communication_costs) + 1), communication_costs, color='green')
                axes[1, 0].set_title(f'Communication Cost per Round - {method_name}')
                axes[1, 0].set_xlabel('Rounds')
                axes[1, 0].set_ylabel('Communication Cost')
                axes[1, 0].grid(True)

            # 4. Résumé final
            axes[1, 1].axis('off')
            summary_text = f"""
            Method: {method_name}
            Final Accuracy: {accuracy_history[-1]:.4f}
            Total Training Time: {sum(round_times):.2f}s
            Avg Time/Round: {sum(round_times) / len(round_times):.2f}s
            Total Communication: {sum(communication_costs)}
            Rounds Completed: {len(accuracy_history)}
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                            verticalalignment='center', fontfamily='monospace')
            axes[1, 1].set_title('Summary')

        else:
            # Structure FL-Comparison (multiples méthodes) - garde l'ancien code
            methods = list(results.keys())
            accuracies = [results[method].get('final_accuracy', 0.0) for method in methods]
            times = [results[method].get('training_time', 0.0) for method in methods]
            comm_costs = [results[method].get('communication_cost', 0) for method in methods]
            # ... reste du code existant

        plt.suptitle(f'FL Hierarchical Results - {getattr(args, "dataset", "Unknown")}', fontsize=16)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphique sauvegardé: {plot_path}")

    except Exception as e:
        print(f"Erreur création graphique: {e}")
        import traceback
        traceback.print_exc()

def jensen_shannon_distance(p, q):
    """Calcul de la distance de Jensen-Shannon entre deux distributions"""
    p = np.array(p) + 1e-10  # Éviter log(0)
    q = np.array(q) + 1e-10
    p = p / p.sum()          # Normalisation
    q = q / q.sum()
    m = 0.5 * (p + q)        # Moyenne
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    return np.sqrt(js_div)


def select_clients_by_samples(client_ids, clients_data, selection_ratio):
    """Sélectionne les clients avec le plus d'échantillons dans un cluster"""
    if selection_ratio >= 1.0:
        return client_ids  # Tous les clients

    # Compter les échantillons par client
    client_samples = []
    for client_id in client_ids:
        _, y_train = clients_data[client_id]
        num_samples = len(y_train)
        client_samples.append((client_id, num_samples))

    # Trier par nombre d'échantillons (décroissant)
    client_samples.sort(key=lambda x: x[1], reverse=True)

    # Sélectionner le top %
    num_to_select = max(1, int(len(client_ids) * selection_ratio))
    selected = [client_id for client_id, _ in client_samples[:num_to_select]]

    return selected

def save_client_stats_csv(clients, args):
    """
    Save client statistics to CSV file

    Args:
        clients: List of FederatedClient objects
        args: Experiment arguments

    Returns:
        filepath: Path to saved CSV file
    """
    # Create directory
    stats_dir = os.path.join('results', 'client_stats')
    os.makedirs(stats_dir, exist_ok=True)

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"client_stats_{args.dataset}_{args.hierarchy_type}_{timestamp}.csv"
    filepath = os.path.join(stats_dir, filename)

    # Collect statistics
    all_stats = [client.get_client_stats() for client in clients]

    if not all_stats:
        print("Warning: No client statistics to save")
        return None

    # Write CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        # Get all fieldnames from first client
        fieldnames = list(all_stats[0].keys())

        # Add metadata columns
        fieldnames.extend(['experiment_type', 'dataset', 'total_rounds',
                           'local_epochs', 'iid'])

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Write each client's stats
        for stats in all_stats:
            # Add metadata
            stats['experiment_type'] = args.hierarchy_type
            stats['dataset'] = args.dataset
            stats['total_rounds'] = args.rounds
            stats['local_epochs'] = args.local_epochs
            stats['iid'] = args.iid

            writer.writerow(stats)

    print(f"Client statistics saved to: {filepath}")
    return filepath