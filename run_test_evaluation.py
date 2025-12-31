"""
Evaluate train-optimised PPA parameters on test period data.

This script:
1. Loads optimised parameters from train period results
2. Loads test period data (2022-2025)
3. Evaluates each algorithm's parameters on test data
4. Saves results to results/tables/optimisation_*_test.csv

Usage:
    python run_test_evaluation.py

This should be run AFTER train optimisation:
    python run_optimisation.py --period train
    python run_test_evaluation.py
"""

import pandas as pd
from pathlib import Path
import sys

from src.data_preparation import filter_by_period
from src.optimisation_problem import (
    calculate_quantile_bounds,
    validate_all_constraints,
    calculate_penalty
)
from src.ppa_risk_metrics import evaluate_for_optimisation
from src.search_tracking import log_evaluation


def load_train_parameters(algorithm: str, results_dir: Path) -> dict:
    """
    Load optimised parameters from train period results.
    
    Args:
        algorithm: Algorithm name ('differential_evolution', 'dual_annealing', 'shgo')
        results_dir: Directory containing train results
        
    Returns:
        Dictionary with floor, strike, cap parameters
    """
    train_file = results_dir / f'optimisation_{algorithm}_train.csv'
    
    if not train_file.exists():
        raise FileNotFoundError(
            f"Train results not found: {train_file}\n"
            f"Please run: python run_optimisation.py --period train"
        )
    
    df = pd.read_csv(train_file)
    
    # Get the last row (final optimised parameters)
    params = df.iloc[-1]
    
    return {
        'floor': params['floor'],
        'strike': params['strike'],
        'cap': params['cap'],
        'algorithm': algorithm
    }


def evaluate_on_test(
    params: dict,
    prices: pd.Series,
    generation: pd.Series,
    lambda_param: float = 0.1,
    beta: float = 0.05,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'ME'
) -> dict:
    """
    Evaluate fixed parameters on test data.
    
    Args:
        params: Dictionary with floor, strike, cap, algorithm
        prices: Test period prices
        generation: Test period generation
        lambda_param: Risk aversion parameter
        beta: Fairness threshold
        confidence_level: CVaR confidence
        aggregation_freq: Revenue aggregation frequency
        
    Returns:
        Dictionary with all evaluation metrics
    """
    floor = params['floor']
    strike = params['strike']
    cap = params['cap']
    algorithm = params['algorithm']
    
    # Calculate bounds from test data prices (for validation)
    bounds = calculate_quantile_bounds(prices)
    
    # Validate constraints
    is_feasible, struct_viol, fair_viol = validate_all_constraints(
        floor=floor,
        strike=strike,
        cap=cap,
        prices=prices,
        generation=generation,
        bounds=bounds,
        beta=beta,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq
    )
    
    # Calculate penalty
    penalty = calculate_penalty(struct_viol, fair_viol)
    
    # Get metrics
    metrics = evaluate_for_optimisation(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq
    )
    
    # Calculate fitness (same formula as optimisation)
    # Formula: max E[R] - λ·(E[R] - CVaR) = (1-λ)·E[R] + λ·CVaR
    shortfall = metrics['mean_revenue'] - metrics['cvar']
    base_objective = metrics['mean_revenue'] - lambda_param * shortfall
    fitness = base_objective - penalty
    
    # Net transfer percentage
    net_transfer_pct = (metrics['net_transfer'] / metrics['market_baseline']) * 100
    
    # Separate feasibility flags
    structural_feasible = (sum(struct_viol.values()) == 0)
    fairness_feasible = (fair_viol['net_transfer'] <= 0)
    
    return {
        'algorithm': algorithm,
        'floor': floor,
        'strike': strike,
        'cap': cap,
        'spread': cap - floor,
        'fitness': fitness,
        'mean_revenue': metrics['mean_revenue'],
        'cvar': metrics['cvar'],
        'var': metrics['var'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'net_transfer': metrics['net_transfer'],
        'net_transfer_pct': net_transfer_pct,
        'market_baseline': metrics['market_baseline'],
        'is_feasible': is_feasible,
        'structural_feasible': structural_feasible,
        'fairness_feasible': fairness_feasible,
        'penalty': penalty,
        'lambda': lambda_param,
        'beta': beta
    }


def main():
    workspace_root = Path(__file__).parent
    results_dir = workspace_root / 'results' / 'tables'
    data_dir = workspace_root / 'data_processed'
    
    print("="*80)
    print("TEST PERIOD EVALUATION")
    print("="*80)
    print()
    
    # Load test data
    print("Loading test period data (2022-2025)...")
    df_prices = pd.read_csv(data_dir / 'ES_prices_hourly.csv', index_col=0, parse_dates=True)
    df_gen = pd.read_csv(data_dir / 'ES_generation_hourly_solar.csv', index_col=0, parse_dates=True)
    
    prices = df_prices.iloc[:, 0]
    generation = df_gen.iloc[:, 0]
    
    # Align indices
    common_index = prices.index.intersection(generation.index)
    prices = prices.loc[common_index]
    generation = generation.loc[common_index]
    
    # Filter to test period
    prices_test, generation_test = filter_by_period(prices, generation, 'test')
    
    print(f"Test period: 2022-2025")
    print(f"Records: {len(prices_test):,} hours")
    print(f"Price mean: {prices_test.mean():.2f} EUR/MWh")
    print(f"Price std: {prices_test.std():.2f} EUR/MWh")
    print(f"Price range: [{prices_test.min():.2f}, {prices_test.max():.2f}] EUR/MWh")
    print()
    
    # Algorithms to evaluate
    algorithms = ['differential_evolution', 'dual_annealing', 'shgo']
    
    print("="*80)
    print("EVALUATING TRAIN PARAMETERS ON TEST DATA")
    print("="*80)
    print()
    
    results = []
    
    for algo in algorithms:
        try:
            print(f"\n{algo.upper()}:")
            print("-" * 80)
            
            # Load train parameters
            params = load_train_parameters(algo, results_dir)
            print(f"Train parameters: F={params['floor']:.2f}, K={params['strike']:.2f}, C={params['cap']:.2f}")
            
            # Evaluate on test
            result = evaluate_on_test(
                params=params,
                prices=prices_test,
                generation=generation_test,
                lambda_param=0.1,
                beta=0.05
            )
            
            results.append(result)
            
            print(f"Test fitness: {result['fitness']:,.0f} EUR")
            print(f"Mean revenue: {result['mean_revenue']:,.0f} EUR")
            print(f"CVaR: {result['cvar']:,.0f} EUR")
            print(f"Penalty: {result['penalty']:,.0f} EUR")
            print(f"Feasible: {result['is_feasible']} (structural={result['structural_feasible']}, fairness={result['fairness_feasible']})")
            print(f"Net transfer: {result['net_transfer_pct']:.2f}%")
            
            # Save individual test result
            log_evaluation(
                floor=result['floor'],
                strike=result['strike'],
                cap=result['cap'],
                fitness=result['fitness'],
                mean_revenue=result['mean_revenue'],
                cvar=result['cvar'],
                net_transfer_pct=result['net_transfer_pct'],
                is_feasible=result['is_feasible'],
                search_type='optimisation',
                algorithm=algo,
                lambda_param=result['lambda'],
                beta=result['beta'],
                penalty=result['penalty'],
                structural_feasible=result['structural_feasible'],
                fairness_feasible=result['fairness_feasible'],
                period='test',
                notes='Test period evaluation of train-optimised parameters'
            )
            
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue
        except Exception as e:
            print(f"ERROR: {algo} evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if results:
        print("\n" + "="*80)
        print("TEST EVALUATION SUMMARY")
        print("="*80)
        
        df_summary = pd.DataFrame(results)
        print(df_summary[['algorithm', 'fitness', 'mean_revenue', 'cvar', 'penalty', 'is_feasible']].to_string(index=False))
        print()
        
        print("Results saved to:")
        for algo in algorithms:
            if any(r['algorithm'] == algo for r in results):
                print(f"  - results/tables/optimisation_{algo}_test.csv")
        print()
        
        print("Next step: Analyze results in notebook 04_ES_algorithm_evaluation.ipynb")
    else:
        print("\nERROR: No results generated. Check error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
