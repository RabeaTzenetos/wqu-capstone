"""
Run optimisation experiments comparing scipy algorithms for PPA parameter selection.

This script compares all 3 scipy algorithms (Differential Evolution, 
Dual Annealing, SHGO) on the same problem and saves results for later analysis.

Usage:
    python run_optimisation.py              # Full period (2015-2025)
    python run_optimisation.py --period train   # Train period (2015-2021)
    python run_optimisation.py --period test    # Test period (2023-2025, excludes 2022)

Results are saved to:
- results/tables/optimisation_*.csv (parameter results with metrics)
- results/convergence/convergence_*.json (iteration-by-iteration fitness)
- results/search_logs/search_log_*.json (ALL evaluations with constraints)

Visualisation should be done in notebooks using the saved data.
"""

import argparse
from pathlib import Path
import pandas as pd

from src.ppa_optimiser import compare_algorithms
from src.data_preparation import filter_by_period


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run PPA optimisation experiments')
    parser.add_argument('--period', type=str, default='full',
                       choices=['full', 'train', 'test'],
                       help='Data period: full (2015-2025), train (2015-2021), test (2023-2025)')
    args = parser.parse_args()
    
    period = args.period
    
    # Load Spanish market data
    print("="*80)
    print(f"LOADING SPANISH MARKET DATA ({period.upper()} PERIOD)")
    print("="*80)
    
    data_dir = Path(__file__).parent / 'data_processed'
    df_prices = pd.read_csv(data_dir / 'ES_prices_hourly.csv', index_col=0, parse_dates=True)
    df_gen = pd.read_csv(data_dir / 'ES_generation_hourly_solar.csv', index_col=0, parse_dates=True)
    
    prices = df_prices.iloc[:, 0]
    generation = df_gen.iloc[:, 0]
    
    # Align indices
    common_index = prices.index.intersection(generation.index)
    prices = prices.loc[common_index]
    generation = generation.loc[common_index]
    
    # Filter by period
    prices, generation = filter_by_period(prices, generation, period)
    
    print(f"Period: {period}")
    if period == 'full':
        print(f"  Range: 2015-2025 (all data)")
    elif period == 'train':
        print(f"  Range: 2015-2021 (84 months, CVaR uses 4.2 worst)")
    elif period == 'test':
        print(f"  Range: 2023-2025 (36 months, CVaR uses 1.8 worst, excludes 2022 stress)")
    print(f"Records: {len(prices):,} hours")
    print(f"Price mean: {prices.mean():.2f} EUR/MWh")
    print()
    
    # Baseline reference
    baseline_fitness = 572842  # Best outcome from manual exploration (test_collar_spreads.py)
    print("="*80)
    print("BASELINE REFERENCE")
    print("="*80)
    print("Manual solution: Tight-50")
    print("  F=45.00, K=70.00, C=95.00")
    print(f"  Fitness: {baseline_fitness:,} EUR (λ=0.1, β=5%)")
    print()
    
    # Compare all 3 scipy algorithms
    print("="*80)
    print("COMPARING SCIPY ALGORITHMS")
    print("="*80)
    print()
    
    summary_df, results_dict = compare_algorithms(
        prices=prices,
        generation=generation,
        algorithms=['differential_evolution', 'dual_annealing', 'shgo'],
        lambda_param=0.1,
        beta=0.05,
        confidence_level=0.95,
        aggregation_freq='ME',
        max_iterations=100,
        population_size=15,
        seed=42,
        period=period
    )
    
    # Validation checks
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    for i, row in summary_df.iterrows():
        algo = row['algorithm']
        fitness = row['fitness']
        improvement = ((fitness - baseline_fitness) / baseline_fitness) * 100
        
        print(f"\n{algo}:")
        print(f"  Fitness: {fitness:>12,.0f} EUR")
        print(f"  vs Baseline: {improvement:>8.2f}%")
        print(f"  Time: {row['time_sec']:>11.2f}s")
        print(f"  Evaluations: {row['evaluations']:>8,}")
        print(f"  Spread: {row['spread']:>11.2f} EUR/MWh")
        print(f"  Feasible: {row['is_feasible']}")
        
        if fitness >= baseline_fitness:
            print(f"  ✓ Beat or matched baseline")
        else:
            print(f"  ✗ Below baseline (may need tuning)")
    
    # Best algorithm
    best_idx = summary_df['fitness'].idxmax()
    best_algo = summary_df.loc[best_idx, 'algorithm']
    best_fitness = summary_df.loc[best_idx, 'fitness']
    best_time = summary_df.loc[best_idx, 'time_sec']
    
    print("\n" + "="*80)
    print("WINNER")
    print("="*80)
    print(f"Best algorithm: {best_algo}")
    print(f"Best fitness: {best_fitness:,.0f} EUR")
    print(f"Time: {best_time:.2f}s")
    print()
    
    print("\n" + "="*80)
    print("OPTIMISATION EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nResults saved:")
    
    # Adjust file names based on period
    suffix = f"_{period}" if period != 'full' else ""
    
    print("  CSV files (final solutions with full metrics):")
    print(f"    - results/tables/optimisation_differential_evolution{suffix}.csv")
    print(f"    - results/tables/optimisation_dual_annealing{suffix}.csv")
    print(f"    - results/tables/optimisation_shgo{suffix}.csv")
    
    if period == 'full':
        # Full period includes convergence and search logs
        print("\n  JSON files (convergence history - best solution progression):")
        print("    - results/convergence/convergence_differential_evolution_seed42.json")
        print("    - results/convergence/convergence_dual_annealing_seed42.json")
        print("    - results/convergence/convergence_shgo_seed42.json")
        print("\n  JSON files (search logs - ALL evaluations with constraints):")
        print("    - results/search_logs/search_log_differential_evolution_seed42.json")
        print("    - results/search_logs/search_log_dual_annealing_seed42.json")
        print("    - results/search_logs/search_log_shgo_seed42.json")
    elif period == 'train':
        print("\n  JSON files (convergence and search logs):")
        print("    - results/convergence/convergence_*_seed42.json")
        print("    - results/search_logs/search_log_*_seed42.json")
        print("\nNext step: Run test evaluation with 'python run_test_evaluation.py'")
    
    print()


if __name__ == '__main__':
    main()
