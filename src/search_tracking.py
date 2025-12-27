"""
Search space tracking for PPA optimisation.

Unified logging system for all collar parameter evaluations across:
- Manual exploration (test_collar_spreads.py)
- Quantile-based search (test_quantile_solution.py)
- Heuristic optimisation (DE, DA, SHGO in test_algorithm_comparison.py)

Objective: Create search space log show of all tested combinations and their performance.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime


def log_evaluation(
    floor: float,
    strike: float,
    cap: float,
    fitness: float,
    mean_revenue: float,
    cvar: float,
    net_transfer_pct: float,
    is_feasible: bool,
    search_type: Literal['naive_baseline', 'manual_exploration', 'quantile_search', 'optimisation'],
    algorithm: Optional[str] = None,
    iteration: Optional[int] = None,
    seed: Optional[int] = None,
    lambda_param: float = 0.1,
    beta: float = 0.05,
    optimisation_time: Optional[float] = None,
    num_evaluations: Optional[int] = None,
    penalty: Optional[float] = None,
    structural_feasible: Optional[bool] = None,
    fairness_feasible: Optional[bool] = None,
    period: Optional[str] = None,
    notes: Optional[str] = None,
    output_dir: Path = None
) -> None:
    """
    Log a single parameter evaluation to search history file.
    
    Args:
        floor: Floor price (EUR/MWh)
        strike: Strike price (EUR/MWh)
        cap: Cap price (EUR/MWh)
        fitness: Objective function value (EUR)
        mean_revenue: Mean monthly revenue (EUR)
        cvar: CVaR at confidence level (EUR)
        net_transfer_pct: Net transfer as % of baseline
        is_feasible: Whether constraints are satisfied
        search_type: Type of search being performed
        algorithm: Algorithm name (for optimisation runs)
        iteration: Iteration number (for optimisation runs)
        seed: Random seed (for reproducibility)
        lambda_param: Risk aversion parameter
        beta: Fairness constraint threshold
        optimisation_time: Time taken for optimisation (seconds)
        num_evaluations: Number of function evaluations (for optimisation runs)
        penalty: Total penalty applied for constraint violations
        structural_feasible: Whether structural constraints (F≤K≤C, bounds) are satisfied
        fairness_feasible: Whether fairness constraint (net transfer ≤ β) is satisfied
        period: Data period ('full', 'train', 'test') for file naming suffix
        notes: Optional notes about this evaluation
        output_dir: Directory for results (default: workspace_root/results/tables)
    """
    # Determine output directory
    if output_dir is None:
        # Assume this file is in src/, go up one level to workspace root
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'results' / 'tables'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build filename with period suffix if provided
    if search_type == 'optimisation' and algorithm:
        base_filename = f'optimisation_{algorithm}'
        if period and period != 'full':
            filename = f'{base_filename}_{period}.csv'
        else:
            filename = f'{base_filename}.csv'
    else:
        filename_map = {
            'naive_baseline': 'naive_baselines.csv',
            'manual_exploration': 'manual_exploration.csv',
            'quantile_search': 'quantile_search.csv',
            'optimisation': 'optimisation_history.csv'
        }
        filename = filename_map[search_type]
    
    output_file = output_dir / filename
    
    # Create record
    record = {
        'timestamp': datetime.now().isoformat(),
        'search_type': search_type,
        'algorithm': algorithm or '',
        'iteration': iteration if iteration is not None else '',
        'seed': seed if seed is not None else '',
        'floor': floor,
        'strike': strike,
        'cap': cap,
        'spread': cap - floor,
        'fitness': fitness,
        'mean_revenue': mean_revenue,
        'cvar': cvar,
        'net_transfer_pct': net_transfer_pct,
        'is_feasible': is_feasible,
        'structural_feasible': structural_feasible if structural_feasible is not None else '',
        'fairness_feasible': fairness_feasible if fairness_feasible is not None else '',
        'penalty': penalty if penalty is not None else '',
        'lambda': lambda_param,
        'beta': beta,
        'optimisation_time': optimisation_time if optimisation_time is not None else '',
        'num_evaluations': num_evaluations if num_evaluations is not None else '',
        'notes': notes or ''
    }
    
    # Append to CSV (create if doesn't exist)
    df_new = pd.DataFrame([record])
    
    if output_file.exists():
        df_existing = pd.read_csv(output_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_file, index=False)
    else:
        df_new.to_csv(output_file, index=False)


def log_evaluation_batch(
    evaluations: list[dict],
    search_type: Literal['naive_baseline', 'manual_exploration', 'quantile_search', 'optimisation'],
    algorithm: Optional[str] = None,
    lambda_param: float = 0.1,
    beta: float = 0.05,
    output_dir: Path = None
) -> None:
    """
    Log multiple evaluations at once.
    
    Args:
        evaluations: List of dicts, each containing:
            - floor, strike, cap
            - fitness, mean_revenue, cvar, net_transfer_pct, is_feasible
            - Optional: penalty, structural_feasible, fairness_feasible
            - Optional: iteration, seed, optimisation_time, num_evaluations, notes
        search_type: Type of search being performed
        algorithm: Algorithm name (for optimisation runs)
        lambda_param: Risk aversion parameter
        beta: Fairness constraint threshold
        output_dir: Directory for results
    """
    # Determine output directory
    if output_dir is None:
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'results' / 'tables'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output file
    filename_map = {
        'naive_baseline': 'naive_baselines.csv',
        'manual_exploration': 'manual_exploration.csv',
        'quantile_search': 'quantile_search.csv',
        'optimisation': f'optimisation_{algorithm}.csv' if algorithm else 'optimisation_history.csv'
    }
    
    output_file = output_dir / filename_map[search_type]
    
    # Add metadata to each record
    timestamp = datetime.now().isoformat()
    records = []
    
    for eval_dict in evaluations:
        record = {
            'timestamp': timestamp,
            'search_type': search_type,
            'algorithm': algorithm or '',
            'iteration': eval_dict.get('iteration', ''),
            'seed': eval_dict.get('seed', ''),
            'floor': eval_dict['floor'],
            'strike': eval_dict['strike'],
            'cap': eval_dict['cap'],
            'spread': eval_dict['cap'] - eval_dict['floor'],
            'fitness': eval_dict['fitness'],
            'mean_revenue': eval_dict['mean_revenue'],
            'cvar': eval_dict['cvar'],
            'net_transfer_pct': eval_dict['net_transfer_pct'],
            'is_feasible': eval_dict['is_feasible'],
            'structural_feasible': eval_dict.get('structural_feasible', ''),
            'fairness_feasible': eval_dict.get('fairness_feasible', ''),
            'penalty': eval_dict.get('penalty', ''),
            'lambda': lambda_param,
            'beta': beta,
            'optimisation_time': eval_dict.get('optimisation_time', ''),
            'num_evaluations': eval_dict.get('num_evaluations', ''),
            'notes': eval_dict.get('notes', '')
        }
        records.append(record)
    
    # Append to CSV
    df_new = pd.DataFrame(records)
    
    if output_file.exists():
        df_existing = pd.read_csv(output_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_file, index=False)
    else:
        df_new.to_csv(output_file, index=False)


def get_search_summary(
    search_type: Optional[Literal['naive_baseline', 'manual_exploration', 'quantile_search', 'optimisation']] = None,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Load and summarise search history.
    
    Args:
        search_type: Filter by search type (None = all types)
        output_dir: Directory containing results
        
    Returns:
        DataFrame with search history
    """
    if output_dir is None:
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'results' / 'tables'
    
    if search_type is None:
        # Load all search files
        patterns = [
            'naive_baselines.csv',
            'manual_exploration.csv',
            'quantile_search.csv',
            'optimisation_*.csv'
        ]
        
        dfs = []
        for pattern in patterns:
            for file in output_dir.glob(pattern):
                if file.exists():
                    dfs.append(pd.read_csv(file))
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        # Load specific search type
        filename_map = {
            'naive_baseline': 'naive_baselines.csv',
            'manual_exploration': 'manual_exploration.csv',
            'quantile_search': 'quantile_search.csv',
            'optimisation': 'optimisation_*.csv'  # Will need to handle glob
        }
        
        if search_type == 'optimisation':
            # Load all optimisation files
            dfs = []
            for file in output_dir.glob('optimisation_*.csv'):
                if file.exists():
                    dfs.append(pd.read_csv(file))
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            file_path = output_dir / filename_map[search_type]
            if file_path.exists():
                return pd.read_csv(file_path)
            else:
                return pd.DataFrame()


def clear_search_history(
    search_type: Optional[Literal['naive_baseline', 'manual_exploration', 'quantile_search', 'optimisation']] = None,
    output_dir: Path = None
) -> None:
    """
    Clear search history files (useful for fresh starts).
    
    Args:
        search_type: Type to clear (None = clear all)
        output_dir: Directory containing results
    """
    if output_dir is None:
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'results' / 'tables'
    
    if search_type is None:
        # Clear all
        patterns = [
            'naive_baselines.csv',
            'manual_exploration.csv', 
            'quantile_search.csv',
            'optimisation_*.csv'
        ]
        for pattern in patterns:
            for file in output_dir.glob(pattern):
                file.unlink()
    else:
        filename_map = {
            'naive_baseline': 'naive_baselines.csv',
            'manual_exploration': 'manual_exploration.csv',
            'quantile_search': 'quantile_search.csv',
            'optimisation': 'optimisation_*.csv'
        }
        
        if search_type == 'optimisation':
            for file in output_dir.glob('optimisation_*.csv'):
                file.unlink()
        else:
            file_path = output_dir / filename_map[search_type]
            if file_path.exists():
                file_path.unlink()


def save_convergence_history(
    algorithm: str,
    convergence_history: list,
    seed: Optional[int] = None,
    period: Optional[str] = None,
    output_dir: Path = None
) -> None:
    """
    Save convergence history to JSON file for later visualisation.
    
    Args:
        algorithm: Algorithm name (e.g., 'differential_evolution')
        convergence_history: List of fitness values at each iteration
        seed: Random seed used (for filename)
        period: Data period ('full', 'train', 'test') for filename suffix
        output_dir: Directory for results (default: workspace_root/results/convergence)
    """
    if output_dir is None:
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'results' / 'convergence'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with seed and period if provided
    filename = f'convergence_{algorithm}'
    if seed is not None:
        filename += f'_seed{seed}'
    if period and period != 'full':
        filename += f'_{period}'
    filename += '.json'
    
    output_file = output_dir / filename
    
    # Save as JSON
    data = {
        'algorithm': algorithm,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'convergence_history': convergence_history
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_convergence_history(
    algorithm: str,
    seed: Optional[int] = None,
    output_dir: Path = None
) -> Optional[list]:
    """
    Load convergence history from JSON file.
    
    Args:
        algorithm: Algorithm name
        seed: Random seed (if specified when saving)
        output_dir: Directory containing results
        
    Returns:
        List of fitness values or None if file doesn't exist
    """
    if output_dir is None:
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'results' / 'convergence'
    
    filename = f'convergence_{algorithm}'
    if seed is not None:
        filename += f'_seed{seed}'
    filename += '.json'
    
    file_path = output_dir / filename
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['convergence_history']
    else:
        return None


def save_search_log(
    algorithm: str,
    search_log: list,
    seed: Optional[int] = None,
    period: Optional[str] = None,
    output_dir: Path = None
) -> None:
    """
    Save complete search log (ALL evaluations) to JSON file.
    
    This captures every single evaluation during optimisation, not just best solutions.
    Useful for detailed search space analysis and comparing algorithm exploration patterns.
    
    Args:
        algorithm: Algorithm name
        search_log: List of dicts with all evaluation details
        seed: Random seed (if specified)
        period: Data period ('full', 'train', 'test') for filename suffix
        output_dir: Directory for results (default: results/search_logs/)
    
    Example:
        >>> save_search_log(
        ...     algorithm='differential_evolution',
        ...     search_log=all_evaluations,
        ...     seed=42,
        ...     period='train'
        ... )
        # Creates: results/search_logs/search_log_differential_evolution_seed42_train.json
    """
    if output_dir is None:
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'results' / 'search_logs'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f'search_log_{algorithm}'
    if seed is not None:
        filename += f'_seed{seed}'
    if period and period != 'full':
        filename += f'_{period}'
    filename += '.json'
    
    output_file = output_dir / filename
    
    data = {
        'algorithm': algorithm,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        'total_evaluations': len(search_log),
        'search_log': search_log
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Search log saved: {output_file} ({len(search_log)} evaluations)")


def load_search_log(
    algorithm: str,
    seed: Optional[int] = None,
    output_dir: Path = None
) -> Optional[list]:
    """
    Load complete search log from JSON file.
    
    Args:
        algorithm: Algorithm name
        seed: Random seed (if specified when saving)
        output_dir: Directory containing results
        
    Returns:
        List of all evaluations or None if file doesn't exist
    """
    if output_dir is None:
        workspace_root = Path(__file__).parent.parent
        output_dir = workspace_root / 'results' / 'search_logs'
    
    filename = f'search_log_{algorithm}'
    if seed is not None:
        filename += f'_seed{seed}'
    filename += '.json'
    
    file_path = output_dir / filename
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['search_log']
    else:
        return None
