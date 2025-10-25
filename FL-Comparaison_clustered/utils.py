"""
Utility functions for FL Comparison Framework
"""

import os
import csv
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt


def save_results(results, args, method_name):
    """
    Save training results to file
    
    Args:
        results: Dictionary with training metrics
        args: Command line arguments
        method_name: Name of the FL method
    """
    # Create results directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{args.dataset}_{method_name}_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    # Write results
    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"FL COMPARISON RESULTS - {method_name.upper()}\n")
        f.write("="*60 + "\n\n")
        
        # Configuration
        f.write("Configuration:\n")
        f.write(f"  Dataset: {args.dataset}\n")
        f.write(f"  Method: {method_name}\n")
        f.write(f"  Clients: {args.clients}\n")
        f.write(f"  Rounds: {args.rounds}\n")
        f.write(f"  Local Epochs: {args.local_epochs}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  IID: {args.iid}\n")
        
        if method_name == 'agglomerative':
            f.write(f"  JS Threshold: {args.js_threshold}\n")
            f.write(f"  Selection Ratio: {args.selection_ratio}\n")
            f.write(f"  Subset Mode: {args.subset_mode}\n")
            if 'num_clusters' in results:
                f.write(f"  Clusters Created: {results['num_clusters']}\n")
        
        if method_name == 'fedprox':
            f.write(f"  Mu: {args.mu}\n")
        
        if method_name == 'ifca':
            f.write(f"  Num Clusters: {args.num_clusters}\n")
        
        f.write("\n")
        
        # Results summary
        f.write("Results:\n")
        f.write(f"  Final Accuracy: {results['accuracy_history'][-1]:.4f}\n")
        f.write(f"  Final Loss: {results['loss_history'][-1]:.4f}\n")
        f.write(f"  Total Time: {sum(results['round_times']):.2f}s\n")
        f.write(f"  Avg Time/Round: {np.mean(results['round_times']):.2f}s\n")
        f.write("\n")
        
        # Round-by-round results
        f.write("Round-by-Round Results:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Round':<8} {'Accuracy':<12} {'Loss':<12} {'Time(s)':<10}\n")
        f.write("-"*60 + "\n")
        
        for i in range(len(results['accuracy_history'])):
            f.write(f"{i+1:<8} {results['accuracy_history'][i]:<12.4f} "
                   f"{results['loss_history'][i]:<12.4f} "
                   f"{results['round_times'][i]:<10.2f}\n")
        
        f.write("="*60 + "\n")
    
    print(f"Results saved to: {filepath}")
    
    # Also save as JSON for easier processing
    json_filename = filename.replace('.txt', '.json')
    json_filepath = os.path.join(results_dir, json_filename)
    
    # Convert numpy types to Python types for JSON serialization
    json_results = {
        'config': {
            'dataset': args.dataset,
            'method': method_name,
            'clients': args.clients,
            'rounds': args.rounds,
            'local_epochs': args.local_epochs,
            'batch_size': args.batch_size,
            'iid': args.iid
        },
        'results': {
            'accuracy_history': [float(x) for x in results['accuracy_history']],
            'loss_history': [float(x) for x in results['loss_history']],
            'round_times': [float(x) for x in results['round_times']],
            'final_accuracy': float(results['accuracy_history'][-1]),
            'final_loss': float(results['loss_history'][-1]),
            'total_time': float(sum(results['round_times'])),
            'avg_time_per_round': float(np.mean(results['round_times']))
        }
    }
    
    with open(json_filepath, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"JSON results saved to: {json_filepath}")


def save_client_stats_csv(clients, args, method_name):
    """
    Save client statistics to CSV file
    
    Args:
        clients: List of FederatedClient objects
        args: Command line arguments
        method_name: Name of the FL method
    """
    # Create results directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"client_stats_{args.dataset}_{method_name}_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    
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
        fieldnames.extend(['method', 'dataset', 'total_rounds',
                          'local_epochs', 'iid'])
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write each client's stats
        for stats in all_stats:
            # Add metadata
            stats['method'] = method_name
            stats['dataset'] = args.dataset
            stats['total_rounds'] = args.rounds
            stats['local_epochs'] = args.local_epochs
            stats['iid'] = args.iid
            
            writer.writerow(stats)
    
    print(f"Client statistics saved to: {filepath}")
    return filepath


def plot_comparison(all_results, args):
    """
    Plot comparison of all methods
    
    Args:
        all_results: Dictionary mapping method names to results
        args: Command line arguments
    """
    # Create visualizations directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'FL Methods Comparison - {args.dataset.upper()}', fontsize=16)
    
    # Plot 1: Accuracy over rounds
    ax1 = axes[0, 0]
    for method, results in all_results.items():
        ax1.plot(results['accuracy_history'], label=method.upper(), marker='o', markersize=3)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Accuracy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss over rounds
    ax2 = axes[0, 1]
    for method, results in all_results.items():
        ax2.plot(results['loss_history'], label=method.upper(), marker='o', markersize=3)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Test Loss Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time per round
    ax3 = axes[1, 0]
    for method, results in all_results.items():
        ax3.plot(results['round_times'], label=method.upper(), marker='o', markersize=3)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Training Time per Round')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary bar chart
    ax4 = axes[1, 1]
    methods = list(all_results.keys())
    final_accs = [results['accuracy_history'][-1] for results in all_results.values()]
    avg_times = [np.mean(results['round_times']) for results in all_results.values()]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x - width/2, final_accs, width, label='Final Accuracy', alpha=0.8)
    bars2 = ax4_twin.bar(x + width/2, avg_times, width, label='Avg Time/Round', alpha=0.8, color='orange')
    
    ax4.set_xlabel('Method')
    ax4.set_ylabel('Accuracy', color='blue')
    ax4_twin.set_ylabel('Time (seconds)', color='orange')
    ax4.set_title('Final Performance Summary')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.upper() for m in methods], rotation=45)
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_{args.dataset}_{timestamp}.png"
    filepath = os.path.join(viz_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {filepath}")
    
    plt.close()


def create_summary_table(all_results):
    """
    Create a summary table of all results
    
    Args:
        all_results: Dictionary mapping method names to results
    
    Returns:
        Summary string
    """
    summary = "\n" + "="*80 + "\n"
    summary += "SUMMARY TABLE\n"
    summary += "="*80 + "\n"
    summary += f"{'Method':<20} {'Final Acc':<12} {'Avg Time/Round':<15} {'Total Time':<12}\n"
    summary += "-"*80 + "\n"
    
    for method, results in all_results.items():
        final_acc = results['accuracy_history'][-1]
        avg_time = np.mean(results['round_times'])
        total_time = sum(results['round_times'])
        
        summary += f"{method.upper():<20} {final_acc:<12.4f} {avg_time:<15.2f} {total_time:<12.2f}\n"
    
    summary += "="*80 + "\n"
    
    return summary