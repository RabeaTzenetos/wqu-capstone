"""
Results analysis module for PPA optimisation evaluation.

This module provides clean functions for loading, comparing, and analyzing
optimisation results across different algorithms, periods, and stress scenarios.

Designed to keep notebooks minimal: load → function call → plot.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.optimisation_problem import (
    calculate_quantile_bounds,
    validate_all_constraints,
    calculate_penalty
)
from src.ppa_risk_metrics import evaluate_for_optimisation


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def load_optimisation_results(
    algorithms: List[str] = ['differential_evolution', 'dual_annealing', 'shgo'],
    periods: List[str] = ['full', 'train', 'test'],
    results_dir: Path = None
) -> Dict[str, pd.DataFrame]:
    """
    Load optimisation result CSVs for multiple algorithms and periods.
    
    Args:
        algorithms: List of algorithm names
        periods: List of periods ('full', 'train', 'test')
        results_dir: Directory containing results (default: results/tables/)
    
    Returns:
        Dictionary with keys like 'differential_evolution_full',
        'dual_annealing_train', etc. Values are DataFrames.
        
    Example:
        >>> results = load_optimisation_results()
        >>> df_de_full = results['differential_evolution_full']
        >>> df_de_train = results['differential_evolution_train']
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / 'results' / 'tables'
    
    results = {}
    
    for algo in algorithms:
        for period in periods:
            # Construct filename
            if period == 'full':
                filename = f'optimisation_{algo}.csv'
            else:
                filename = f'optimisation_{algo}_{period}.csv'
            
            filepath = results_dir / filename
            
            if filepath.exists():
                results[f'{algo}_{period}'] = pd.read_csv(filepath)
            else:
                print(f"Warning: {filename} not found")
    
    return results


def load_convergence_histories(
    algorithms: List[str] = ['differential_evolution', 'dual_annealing', 'shgo'],
    periods: List[str] = ['full', 'train'],
    seed: int = 42,
    results_dir: Path = None
) -> Dict[str, List]:
    """
    Load convergence history JSONs for multiple algorithms and periods.
    
    Args:
        algorithms: List of algorithm names
        periods: List of periods (typically 'full' and 'train')
        seed: Random seed used in optimisation
        results_dir: Directory containing results (default: results/convergence/)
    
    Returns:
        Dictionary with keys like 'differential_evolution_full',
        values are lists of fitness values at each iteration.
        
    Example:
        >>> histories = load_convergence_histories()
        >>> de_full_history = histories['differential_evolution_full']
        >>> # Plot convergence: plt.plot(de_full_history)
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / 'results' / 'convergence'
    
    histories = {}
    
    for algo in algorithms:
        for period in periods:
            # Construct filename
            if period == 'full':
                filename = f'convergence_{algo}_seed{seed}.json'
            else:
                filename = f'convergence_{algo}_seed{seed}_{period}.json'
            
            filepath = results_dir / filename
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                histories[f'{algo}_{period}'] = data['convergence_history']
            else:
                print(f"Warning: {filename} not found")
    
    return histories


def load_search_logs(
    algorithms: List[str] = ['differential_evolution', 'dual_annealing', 'shgo'],
    periods: List[str] = ['full', 'train'],
    seed: int = 42,
    results_dir: Path = None
) -> Dict[str, List]:
    """
    Load complete search logs (all evaluations) for analysis.
    
    Args:
        algorithms: List of algorithm names
        periods: List of periods (typically 'full' and 'train')
        seed: Random seed used in optimisation
        results_dir: Directory containing results (default: results/search_logs/)
    
    Returns:
        Dictionary with keys like 'differential_evolution_full',
        values are lists of evaluation dicts.
        
    Example:
        >>> logs = load_search_logs()
        >>> de_full_log = logs['differential_evolution_full']
        >>> len(de_full_log)  # Number of evaluations
    """
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / 'results' / 'search_logs'
    
    logs = {}
    
    for algo in algorithms:
        for period in periods:
            # Construct filename
            if period == 'full':
                filename = f'search_log_{algo}_seed{seed}.json'
            else:
                filename = f'search_log_{algo}_seed{seed}_{period}.json'
            
            filepath = results_dir / filename
            
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logs[f'{algo}_{period}'] = data['search_log']
            else:
                print(f"Warning: {filename} not found")
    
    return logs


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_train_test(
    results: Dict[str, pd.DataFrame],
    algorithms: List[str] = ['differential_evolution', 'dual_annealing', 'shgo']
) -> pd.DataFrame:
    """
    Compare train vs test performance for all algorithms.
    
    Calculates degradation metrics and component breakdown.
    
    Args:
        results: Dictionary from load_optimisation_results()
        algorithms: List of algorithm names to compare
    
    Returns:
        DataFrame with columns:
        - algorithm, period, fitness, mean_revenue, cvar, penalty
        - fitness_change (test - train)
        - fitness_change_pct (%)
        - component analysis
        
    Example:
        >>> results = load_optimisation_results()
        >>> comparison = compare_train_test(results)
        >>> print(comparison[['algorithm', 'period', 'fitness', 'fitness_change_pct']])
    """
    rows = []
    
    for algo in algorithms:
        train_key = f'{algo}_train'
        test_key = f'{algo}_test'
        
        if train_key not in results or test_key not in results:
            print(f"Warning: Missing train or test results for {algo}")
            continue
        
        # Get last row (final result)
        train_row = results[train_key].iloc[-1]
        test_row = results[test_key].iloc[-1]
        
        # Train metrics
        rows.append({
            'algorithm': algo,
            'period': 'train',
            'fitness': train_row['fitness'],
            'mean_revenue': train_row['mean_revenue'],
            'cvar': train_row['cvar'],
            'penalty': train_row.get('penalty', 0),
            'is_feasible': train_row['is_feasible'],
            'net_transfer_pct': train_row['net_transfer_pct'],
            'floor': train_row['floor'],
            'strike': train_row['strike'],
            'cap': train_row['cap'],
            'spread': train_row['spread']
        })
        
        # Test metrics
        rows.append({
            'algorithm': algo,
            'period': 'test',
            'fitness': test_row['fitness'],
            'mean_revenue': test_row['mean_revenue'],
            'cvar': test_row['cvar'],
            'penalty': test_row.get('penalty', 0),
            'is_feasible': test_row['is_feasible'],
            'net_transfer_pct': test_row['net_transfer_pct'],
            'floor': test_row['floor'],
            'strike': test_row['strike'],
            'cap': test_row['cap'],
            'spread': test_row['spread']
        })
    
    df = pd.DataFrame(rows)
    
    # Calculate changes
    for algo in algorithms:
        train_mask = (df['algorithm'] == algo) & (df['period'] == 'train')
        test_mask = (df['algorithm'] == algo) & (df['period'] == 'test')
        
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            train_fitness = df.loc[train_mask, 'fitness'].values[0]
            test_fitness = df.loc[test_mask, 'fitness'].values[0]
            
            change = test_fitness - train_fitness
            change_pct = (change / train_fitness) * 100
            
            df.loc[test_mask, 'fitness_change'] = change
            df.loc[test_mask, 'fitness_change_pct'] = change_pct
    
    return df


def evaluate_by_year(
    floor: float,
    strike: float,
    cap: float,
    prices: pd.Series,
    generation: pd.Series,
    year: int,
    lambda_param: float = 0.1,
    beta: float = 0.05,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'ME'
) -> Dict:
    """
    Evaluate fixed PPA parameters on a specific year's data.
    
    Useful for year-level stress analysis (e.g., 2022 Ukraine crisis).
    
    Args:
        floor, strike, cap: PPA parameters to evaluate
        prices: Full price series
        generation: Full generation series
        year: Year to filter (e.g., 2022)
        lambda_param: Risk aversion parameter
        beta: Fairness constraint threshold
        confidence_level: CVaR confidence level
        aggregation_freq: Revenue aggregation frequency
    
    Returns:
        Dictionary with fitness, metrics, and constraint info for that year
        
    Example:
        >>> # Evaluate train params on 2022
        >>> result_2022 = evaluate_by_year(
        ...     floor=39.10, strike=60.36, cap=172.43,
        ...     prices=prices, generation=gen, year=2022
        ... )
        >>> print(f"2022 fitness: {result_2022['fitness']:.0f}")
    """
    # Filter to specific year
    year_mask = prices.index.year == year
    prices_year = prices[year_mask]
    gen_year = generation[year_mask]
    
    if len(prices_year) == 0:
        raise ValueError(f"No data found for year {year}")
    
    # Calculate bounds (from full or year data?)
    bounds = calculate_quantile_bounds(prices_year)
    
    # Validate constraints
    is_feasible, struct_viol, fair_viol = validate_all_constraints(
        floor=floor,
        strike=strike,
        cap=cap,
        prices=prices_year,
        generation=gen_year,
        bounds=bounds,
        beta=beta,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq
    )
    
    # Calculate penalty (handle None case and NaN values)
    if struct_viol is None or fair_viol is None:
        penalty = 0.0
        is_feasible = False
    else:
        # Calculate penalty, treating NaN values as 0
        struct_penalty = sum(v for v in struct_viol.values() if not pd.isna(v))
        fair_penalty = sum(v for v in fair_viol.values() if not pd.isna(v))
        penalty = struct_penalty * 1e8 + fair_penalty * 10.0
    
    # Get metrics
    metrics = evaluate_for_optimisation(
        prices=prices_year,
        generation=gen_year,
        floor=floor,
        strike=strike,
        cap=cap,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq
    )
    
    # Calculate fitness (handle None/NaN cases)
    if metrics is None or 'mean_revenue' not in metrics or 'cvar' not in metrics:
        # Return error state
        return {
            'year': year,
            'floor': floor,
            'strike': strike,
            'cap': cap,
            'spread': cap - floor,
            'fitness': np.nan,
            'mean_revenue': np.nan,
            'cvar': np.nan,
            'var': np.nan,
            'sharpe_ratio': np.nan,
            'net_transfer': np.nan,
            'net_transfer_pct': np.nan,
            'market_baseline': np.nan,
            'penalty': np.nan,
            'is_feasible': False,
            'structural_feasible': False,
            'fairness_feasible': False,
            'num_hours': len(prices_year)
        }
    
    # Formula: max E[R] - λ·(E[R] - CVaR) = (1-λ)·E[R] + λ·CVaR
    shortfall = metrics['mean_revenue'] - metrics['cvar']
    base_objective = metrics['mean_revenue'] - lambda_param * shortfall
    fitness = base_objective - penalty
    
    # Net transfer percentage
    net_transfer_pct = (metrics['net_transfer'] / metrics['market_baseline']) * 100
    
    # Separate feasibility
    structural_feasible = (sum(struct_viol.values()) == 0)
    fairness_feasible = (fair_viol['net_transfer'] <= 0)
    
    return {
        'year': year,
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
        'penalty': penalty,
        'is_feasible': is_feasible,
        'structural_feasible': structural_feasible,
        'fairness_feasible': fairness_feasible,
        'num_hours': len(prices_year)
    }


def stress_test_by_years(
    train_results: Dict[str, pd.DataFrame],
    prices: pd.Series,
    generation: pd.Series,
    years: List[int] = [2022, 2023, 2024, 2025],
    algorithms: List[str] = ['differential_evolution', 'dual_annealing', 'shgo'],
    **eval_kwargs
) -> pd.DataFrame:
    """
    Evaluate train-optimised parameters across multiple years.
    
    Creates a comprehensive stress test analysis showing year-by-year
    performance for each algorithm's train-optimised parameters.
    
    Args:
        train_results: Dictionary with train results (from load_optimisation_results)
        prices: Full price series
        generation: Full generation series
        years: List of years to evaluate
        algorithms: List of algorithms to analyse
        **eval_kwargs: Additional arguments for evaluate_by_year()
    
    Returns:
        DataFrame with one row per (algorithm, year) combination
        
    Example:
        >>> results = load_optimisation_results()
        >>> stress_df = stress_test_by_years(
        ...     train_results=results,
        ...     prices=prices,
        ...     generation=gen,
        ...     years=[2022, 2023, 2024, 2025]
        ... )
        >>> # Identify worst year for each algorithm
        >>> worst = stress_df.groupby('algorithm')['fitness'].idxmin()
    """
    rows = []
    
    for algo in algorithms:
        train_key = f'{algo}_train'
        
        if train_key not in train_results:
            print(f"Warning: No train results for {algo}")
            continue
        
        # Get train parameters
        train_row = train_results[train_key].iloc[-1]
        floor = train_row['floor']
        strike = train_row['strike']
        cap = train_row['cap']
        
        # Evaluate on each year
        for year in years:
            try:
                result = evaluate_by_year(
                    floor=floor,
                    strike=strike,
                    cap=cap,
                    prices=prices,
                    generation=generation,
                    year=year,
                    **eval_kwargs
                )
                result['algorithm'] = algo
                rows.append(result)
            except Exception as e:
                print(f"Warning: Failed to evaluate {algo} on {year}: {e}")
                continue
    
    df = pd.DataFrame(rows)
    
    # Reorder columns for readability
    cols_first = ['algorithm', 'year', 'fitness', 'mean_revenue', 'cvar', 'penalty', 'is_feasible']
    cols_rest = [c for c in df.columns if c not in cols_first]
    df = df[cols_first + cols_rest]
    
    return df


# ============================================================================
# SUMMARY FUNCTIONS
# ============================================================================

def summarise_full_period(
    results: Dict[str, pd.DataFrame],
    algorithms: List[str] = ['differential_evolution', 'dual_annealing', 'shgo']
) -> pd.DataFrame:
    """
    Create summary table for full-period optimisation results.
    
    Args:
        results: Dictionary from load_optimisation_results()
        algorithms: List of algorithms to summarise
    
    Returns:
        DataFrame with key metrics for each algorithm
        
    Example:
        >>> results = load_optimisation_results()
        >>> summary = summarise_full_period(results)
        >>> print(summary.to_string(index=False))
    """
    rows = []
    
    for algo in algorithms:
        key = f'{algo}_full'
        
        if key not in results:
            print(f"Warning: No full period results for {algo}")
            continue
        
        row = results[key].iloc[-1]
        
        rows.append({
            'algorithm': algo,
            'fitness': row['fitness'],
            'mean_revenue': row['mean_revenue'],
            'cvar': row['cvar'],
            'floor': row['floor'],
            'strike': row['strike'],
            'cap': row['cap'],
            'spread': row['spread'],
            'net_transfer_pct': row['net_transfer_pct'],
            'is_feasible': row['is_feasible'],
            'time_sec': row.get('optimisation_time', np.nan),
            'evaluations': row.get('num_evaluations', np.nan)
        })
    
    df = pd.DataFrame(rows)
    
    # Rank by fitness
    df['fitness_rank'] = df['fitness'].rank(ascending=False).astype(int)
    
    return df


def analyse_convergence_speed(
    histories: Dict[str, List],
    algorithms: List[str] = ['differential_evolution', 'dual_annealing', 'shgo'],
    period: str = 'full'
) -> pd.DataFrame:
    """
    analyse convergence characteristics for each algorithm.
    
    Args:
        histories: Dictionary from load_convergence_histories()
        algorithms: List of algorithms to analyse
        period: Period to analyse ('full' or 'train')
    
    Returns:
        DataFrame with convergence metrics:
        - iterations_to_best: When best solution found
        - final_fitness: Final converged fitness
        - improvement_from_start: Fitness gain from initialization
        
    Example:
        >>> histories = load_convergence_histories()
        >>> conv_analysis = analyse_convergence_speed(histories)
        >>> print(conv_analysis)
    """
    rows = []
    
    for algo in algorithms:
        key = f'{algo}_{period}'
        
        if key not in histories:
            print(f"Warning: No convergence history for {algo}_{period}")
            continue
        
        history = histories[key]
        
        # Extract fitness values (history contains dicts with iteration details)
        # Fitness values in the JSON are already positive (actual objective values)
        if isinstance(history[0], dict):
            fitness_values = [entry['fitness'] for entry in history]
        else:
            # Fallback for simple list of numbers (if they're negative from scipy, negate)
            fitness_values = [-f if f < 0 else f for f in history]
        
        rows.append({
            'algorithm': algo,
            'total_iterations': len(fitness_values),
            'initial_fitness': round(fitness_values[0], 2),
            'final_fitness': round(fitness_values[-1], 2),
            'best_fitness': round(max(fitness_values), 2),
            'iterations_to_best': fitness_values.index(max(fitness_values)) + 1,
            'improvement_from_start': round(max(fitness_values) - fitness_values[0], 2),
            'improvement_pct': round(((max(fitness_values) - fitness_values[0]) / abs(fitness_values[0])) * 100, 2)
        })
    
    return pd.DataFrame(rows)
