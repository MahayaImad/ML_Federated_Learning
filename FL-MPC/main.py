"""
Point d'entrée FL-MPC
"""
import argparse
import sys

from data_preparation import prepare_federated_cifar10

from fl_vs_mpc import compare_fl_vs_mpc, plot_comparison,save_comparison_results


def main():
    parser = argparse.ArgumentParser(description='FL-MPC vs FL Classique')
    parser.add_argument('--iid', action='store_true', help='Distribution IID')
    parser.add_argument('--rounds', type=int, default=30, help='Nombre de rounds')
    parser.add_argument('--plot', action='store_true', help='Afficher graphiques')

    args = parser.parse_args()

    # Préparation des données (même que FL classique)
    fed_data, test_data, _ = prepare_federated_cifar10(iid=args.iid)

    # Comparaison
    results = compare_fl_vs_mpc(fed_data, test_data, args.rounds)

    if args.plot:
        plot_comparison(results)

    save_comparison_results(results, args)


if __name__ == "__main__":
    main()