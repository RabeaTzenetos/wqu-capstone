"""
PPA Optimisation using scipy.optimize.

Provides unified interface to optimise collared PPA parameters using
multiple scipy algorithms. All algorithms are robust, well-tested,
and part of the standard scientific Python stack.

Available algorithms:
- differential_evolution: Population-based evolutionary algorithm (DEFAULT)
  * Excellent for constrained continuous optimisation
  * Robust and handles constraints naturally
  * Recommended as first choice
  
- dual_annealing: Simulated annealing with dual temperature scheme
  * Good for highly non-linear/rugged landscapes
  * Stochastic search with temperature-based exploration
  * May find different local optima than DE
  
- shgo: Simplicial Homology Global Optimisation
  * Deterministic geometry-based global optimisation
  * Uses simplicial complexes to explore search space
  * Provides theoretical convergence guarantees

Key functions:
- optimise_ppa(): Single optimisation run with one algorithm
- optimise_ppa_multirun(): Multiple runs for statistical validation
- compare_algorithms(): Compare multiple algorithms on same problem
- compare_risk_aversion(): Explore Pareto frontier across risk levels
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, OptimizeResult
from typing import Dict, Tuple, Optional, Literal, List
from dataclasses import dataclass
import time

from src.optimisation_problem import (
    objective_function,
    calculate_quantile_bounds,
    validate_all_constraints,
    calculate_penalty
)
from src.ppa_risk_metrics import evaluate_for_optimisation
from src.search_tracking import log_evaluation, save_convergence_history


@dataclass
class OptimisationResult:
    """Results from PPA optimisation."""
    floor: float
    strike: float
    cap: float
    spread: float
    fitness: float
    is_feasible: bool
    net_transfer_pct: float
    mean_revenue: float
    cvar: float
    optimisation_time: float
    num_evaluations: int
    scipy_result: OptimizeResult  # Full scipy result for diagnostics
    convergence_history: list  # Best solution progression (for thesis narrative)
    search_log: list  # ALL evaluations (for search space analysis)


def optimise_ppa(
    prices: pd.Series,
    generation: pd.Series,
    lambda_param: float = 0.1,
    beta: float = 0.05,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'ME',
    algorithm: Literal['differential_evolution', 'dual_annealing', 'shgo'] = 'differential_evolution',
    max_iterations: int = 1000,
    population_size: int = 15,
    seed: Optional[int] = None,
    period: str = 'full',
    verbose: bool = True
) -> OptimisationResult:
    """
    Optimise collared PPA parameters using scipy.optimize.
    
    Maximizes: E[Revenue] - λ·CVaR - penalties
    Subject to: F ≤ K ≤ C, quantile bounds, fairness constraint
    
    Args:
        prices: Historical spot prices (EUR/MWh)
        generation: Historical generation (MWh)
        lambda_param: Risk aversion parameter (0 = risk-neutral, 1 = very risk-averse)
        beta: Fairness constraint threshold (e.g., 0.05 = 5% max transfer)
        confidence_level: CVaR confidence level (default 0.95 = 5% tail)
        aggregation_freq: Revenue aggregation ('ME'=monthly, 'YE'=yearly)
        algorithm: Optimisation algorithm ('differential_evolution' recommended)
        max_iterations: Maximum iterations (default 1000)
        population_size: DE population size multiplier (default 15)
                        Actual population = 15 * num_parameters = 45
        seed: Random seed for reproducibility
        period: Data period ('full', 'train', 'test') for file naming
        verbose: Print progress
        
    Returns:
        OptimisationResult with best parameters and performance metrics
        
    Example:
        >>> result = optimise_ppa(
        ...     prices=prices, 
        ...     generation=generation,
        ...     lambda_param=0.1,
        ...     beta=0.05,
        ...     period='train'
        ... )
        >>> print(f"Optimal: F={result.floor:.2f}, K={result.strike:.2f}, C={result.cap:.2f}")
        >>> print(f"Fitness: {result.fitness:,.0f} EUR")
    """
    
    # Calculate bounds from price quantiles
    bounds_dict = calculate_quantile_bounds(prices)
    
    # scipy expects bounds as list of tuples: [(lower, upper), ...]
    bounds = [
        bounds_dict['floor'],    # Floor bounds
        bounds_dict['strike'],   # Strike bounds
        bounds_dict['cap']       # Cap bounds
    ]
    
    if verbose:
        print("=" * 80)
        print("PPA PARAMETER OPTIMISATION")
        print("=" * 80)
        print(f"Algorithm: scipy.optimize.{algorithm}")
        print(f"Risk aversion (λ): {lambda_param}")
        print(f"Fairness constraint (β): {beta:.1%}")
        print(f"CVaR confidence: {confidence_level:.1%}")
        print(f"Aggregation: {aggregation_freq}")
        print()
        print("Parameter bounds:")
        print(f"  Floor:  [{bounds[0][0]:>6.2f}, {bounds[0][1]:>6.2f}] EUR/MWh")
        print(f"  Strike: [{bounds[1][0]:>6.2f}, {bounds[1][1]:>6.2f}] EUR/MWh")
        print(f"  Cap:    [{bounds[2][0]:>6.2f}, {bounds[2][1]:>6.2f}] EUR/MWh")
        print()
    
    # Define objective function for scipy (minimization)
    # scipy minimizes, so we negate our fitness function (which we want to maximize)
    convergence_history = []  # Track best solution updates (for thesis narrative)
    search_log = []  # Track ALL evaluations (for search space analysis)
    evaluation_count = [0]  # Mutable counter for tracking evaluations
    
    def objective(params):
        floor, strike, cap = params
        fitness = objective_function(
            floor=floor,
            strike=strike,
            cap=cap,
            prices=prices,
            generation=generation,
            bounds=bounds_dict,
            lambda_param=lambda_param,
            beta=beta,
            confidence_level=confidence_level,
            aggregation_freq=aggregation_freq
        )
        
        # Track this evaluation in search log (for search space analysis)
        evaluation_count[0] += 1
        
        # Validate constraints for search log
        is_feasible, struct_viol, fair_viol = validate_all_constraints(
            floor=floor,
            strike=strike,
            cap=cap,
            prices=prices,
            generation=generation,
            bounds=bounds_dict,
            beta=beta,
            confidence_level=confidence_level,
            aggregation_freq=aggregation_freq
        )
        
        structural_feasible = sum(struct_viol.values()) == 0
        fairness_feasible = fair_viol['net_transfer'] <= 0
        penalty = calculate_penalty(
            structural_violations=struct_viol,
            fairness_violations=fair_viol,
            structural_penalty_weight=1e8,
            fairness_penalty_weight=10.0
        )
        
        search_log.append({
            'evaluation': evaluation_count[0],
            'floor': float(floor),
            'strike': float(strike),
            'cap': float(cap),
            'spread': float(cap - floor),
            'fitness': float(fitness),
            'is_feasible': bool(is_feasible),
            'structural_feasible': bool(structural_feasible),
            'fairness_feasible': bool(fairness_feasible),
            'penalty': float(penalty)
        })
        
        return -fitness  # Negate because scipy minimizes
    
    # Callback to track convergence
    def callback(*args, **kwargs):
        """Called after each iteration to track progress.
        
        Handles different callback signatures from scipy algorithms:
        - differential_evolution: callback(xk, convergence=None)
        - dual_annealing: callback(x, f, context)
        - shgo: callback(xk)
        """
        # First argument is always the parameter vector
        xk = args[0]
        floor, strike, cap = xk
        fitness = -objective(xk)  # Convert back to positive
        
        # Validate constraints to get feasibility info
        is_feasible, struct_viol, fair_viol = validate_all_constraints(
            floor=floor,
            strike=strike,
            cap=cap,
            prices=prices,
            generation=generation,
            bounds=bounds_dict,
            beta=beta,
            confidence_level=confidence_level,
            aggregation_freq=aggregation_freq
        )
        
        # Calculate separate feasibility and penalty
        structural_feasible = sum(struct_viol.values()) == 0
        fairness_feasible = fair_viol['net_transfer'] <= 0
        penalty = calculate_penalty(
            structural_violations=struct_viol,
            fairness_violations=fair_viol,
            structural_penalty_weight=1e8,
            fairness_penalty_weight=10.0
        )
        
        convergence_history.append({
            'iteration': len(convergence_history) + 1,
            'floor': float(floor),
            'strike': float(strike),
            'cap': float(cap),
            'spread': float(cap - floor),
            'fitness': float(fitness),
            'is_feasible': bool(is_feasible),
            'structural_feasible': bool(structural_feasible),
            'fairness_feasible': bool(fairness_feasible),
            'penalty': float(penalty)
        })
        return False  # Return False to continue optimisation
    
    # Run optimisation
    start_time = time.time()
    
    if algorithm == 'differential_evolution':
        # Differential Evolution - robust for constrained problems
        scipy_result = differential_evolution(
            func=objective,
            bounds=bounds,
            maxiter=max_iterations,
            popsize=population_size,
            seed=seed,
            workers=1,  # Single core (data isn't picklable for multiprocessing)
            updating='deferred',  # Evaluate entire population before updating
            polish=True,  # Local optimisation at end for fine-tuning
            disp=verbose,
            callback=callback  # Track convergence
        )
    elif algorithm == 'dual_annealing':
        # Simulated Annealing variant - good for non-linear problems
        from scipy.optimize import dual_annealing
        scipy_result = dual_annealing(
            func=objective,
            bounds=bounds,
            maxiter=max_iterations,
            seed=seed,
            callback=callback
        )
    elif algorithm == 'shgo':
        # Simplicial Homology Global Optimisation
        from scipy.optimize import shgo
        scipy_result = shgo(
            func=objective,
            bounds=bounds,
            options={'maxiter': max_iterations},
            callback=callback
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    optimisation_time = time.time() - start_time
    
    # Extract best solution
    floor_opt, strike_opt, cap_opt = scipy_result.x
    fitness_opt = -scipy_result.fun  # Convert back to maximization
    
    # Validate solution
    is_feasible, struct_viol, fair_viol = validate_all_constraints(
        floor=floor_opt,
        strike=strike_opt,
        cap=cap_opt,
        prices=prices,
        generation=generation,
        bounds=bounds_dict,
        beta=beta,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq
    )
    
    # Calculate separate feasibility flags and penalty
    structural_feasible = sum(struct_viol.values()) == 0
    fairness_feasible = fair_viol['net_transfer'] <= 0
    
    # Calculate penalty using same weights as objective function
    penalty = calculate_penalty(
        structural_violations=struct_viol,
        fairness_violations=fair_viol,
        structural_penalty_weight=1e8,
        fairness_penalty_weight=10.0
    )
    
    # Get detailed metrics
    metrics = evaluate_for_optimisation(
        prices=prices,
        generation=generation,
        floor=floor_opt,
        strike=strike_opt,
        cap=cap_opt,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq
    )
    
    net_transfer_pct = (metrics['net_transfer'] / metrics['market_baseline']) * 100
    
    if verbose:
        print()
        print("=" * 80)
        print("OPTIMISATION COMPLETE")
        print("=" * 80)
        print(f"Status: {scipy_result.message}")
        print(f"Iterations: {scipy_result.nit}")
        print(f"Evaluations: {scipy_result.nfev}")
        print(f"Time: {optimisation_time:.2f}s")
        print()
        print("Optimal solution:")
        print(f"  Floor:  {floor_opt:>7.2f} EUR/MWh")
        print(f"  Strike: {strike_opt:>7.2f} EUR/MWh")
        print(f"  Cap:    {cap_opt:>7.2f} EUR/MWh")
        print(f"  Spread: {cap_opt - floor_opt:>7.2f} EUR/MWh")
        print()
        print(f"Performance:")
        print(f"  Fitness:           {fitness_opt:>12,.0f} EUR")
        print(f"  Mean revenue:      {metrics['mean_revenue']:>12,.0f} EUR/month")
        print(f"  CVaR:              {metrics['cvar']:>12,.0f} EUR/month")
        print(f"  Net transfer:      {net_transfer_pct:>12.2f}% (limit: {beta:.1%})")
        print()
        print(f"Feasibility: {is_feasible}")
        if not is_feasible:
            print(f"  Structural violations: {struct_viol}")
            print(f"  Fairness violations: {fair_viol}")
    
    # Create result object
    result = OptimisationResult(
        floor=floor_opt,
        strike=strike_opt,
        cap=cap_opt,
        spread=cap_opt - floor_opt,
        fitness=fitness_opt,
        is_feasible=is_feasible,
        net_transfer_pct=net_transfer_pct,
        mean_revenue=metrics['mean_revenue'],
        cvar=metrics['cvar'],
        optimisation_time=optimisation_time,
        num_evaluations=scipy_result.nfev,
        scipy_result=scipy_result,
        convergence_history=convergence_history,
        search_log=search_log
    )
    
    # Log final result to search tracking system
    if verbose:
        log_evaluation(
            floor=floor_opt,
            strike=strike_opt,
            cap=cap_opt,
            fitness=fitness_opt,
            mean_revenue=metrics['mean_revenue'],
            cvar=metrics['cvar'],
            net_transfer_pct=net_transfer_pct,
            is_feasible=is_feasible,
            search_type='optimisation',
            algorithm=algorithm,
            iteration=scipy_result.nit,
            seed=seed,
            lambda_param=lambda_param,
            beta=beta,
            optimisation_time=optimisation_time,
            num_evaluations=scipy_result.nfev,
            penalty=penalty,
            structural_feasible=structural_feasible,
            fairness_feasible=fairness_feasible,
            period=period,
            notes=f"Converged: {scipy_result.message}"
        )
        
        # Save convergence history for visualisation (best solutions)
        if convergence_history:
            save_convergence_history(
                algorithm=algorithm,
                convergence_history=convergence_history,
                seed=seed,
                period=period
            )
        
        # Save full search log for detailed analysis (ALL evaluations)
        if search_log:
            from src.search_tracking import save_search_log
            save_search_log(
                algorithm=algorithm,
                search_log=search_log,
                seed=seed,
                period=period
            )
    
    return result


def optimise_ppa_multirun(
    prices: pd.Series,
    generation: pd.Series,
    lambda_param: float = 0.1,
    beta: float = 0.05,
    num_runs: int = 3,
    **kwargs
) -> Tuple[OptimisationResult, list]:
    """
    Run optimisation multiple times for statistical validation.
    
    Args:
        prices: Historical spot prices
        generation: Historical generation
        lambda_param: Risk aversion
        beta: Fairness threshold
        num_runs: Number of independent runs
        **kwargs: Additional arguments passed to optimise_ppa()
        
    Returns:
        (best_result, all_results) tuple
    """
    results = []
    
    print("=" * 80)
    print(f"MULTI-RUN OPTIMISATION ({num_runs} independent runs)")
    print("=" * 80)
    print()
    
    for run in range(num_runs):
        print(f"\n{'='*80}")
        print(f"RUN {run + 1} of {num_runs}")
        print(f"{'='*80}\n")
        
        # Use different seed for each run
        seed = kwargs.get('seed', 0) + run if 'seed' in kwargs else run
        kwargs['seed'] = seed
        
        result = optimise_ppa(
            prices=prices,
            generation=generation,
            lambda_param=lambda_param,
            beta=beta,
            **kwargs
        )
        results.append(result)
    
    # Statistical summary
    print("\n" + "=" * 80)
    print("MULTI-RUN SUMMARY")
    print("=" * 80)
    
    fitnesses = [r.fitness for r in results]
    floors = [r.floor for r in results]
    strikes = [r.strike for r in results]
    caps = [r.cap for r in results]
    spreads = [r.spread for r in results]
    
    print(f"\nFitness statistics (n={num_runs}):")
    print(f"  Mean:  {np.mean(fitnesses):>12,.0f} EUR")
    print(f"  Std:   {np.std(fitnesses):>12,.0f} EUR")
    print(f"  CoV:   {(np.std(fitnesses)/np.mean(fitnesses))*100:>12.2f}%")
    print(f"  Range: [{np.min(fitnesses):>10,.0f}, {np.max(fitnesses):>10,.0f}] EUR")
    
    print(f"\nParameter consistency:")
    print(f"  Floor:  {np.mean(floors):>6.2f} ± {np.std(floors):>5.2f} EUR/MWh")
    print(f"  Strike: {np.mean(strikes):>6.2f} ± {np.std(strikes):>5.2f} EUR/MWh")
    print(f"  Cap:    {np.mean(caps):>6.2f} ± {np.std(caps):>5.2f} EUR/MWh")
    print(f"  Spread: {np.mean(spreads):>6.2f} ± {np.std(spreads):>5.2f} EUR/MWh")
    
    # Best result
    best_idx = np.argmax(fitnesses)
    best_result = results[best_idx]
    
    print(f"\nBest solution from Run {best_idx + 1}:")
    print(f"  F={best_result.floor:.2f}, K={best_result.strike:.2f}, "
          f"C={best_result.cap:.2f}")
    print(f"  Fitness: {best_result.fitness:,.0f} EUR")
    print()
    
    return best_result, results


def compare_algorithms(
    prices: pd.Series,
    generation: pd.Series,
    algorithms: list = ['differential_evolution', 'dual_annealing', 'shgo'],
    lambda_param: float = 0.1,
    beta: float = 0.05,
    period: str = 'full',
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, OptimisationResult]]:
    """
    Compare multiple scipy.optimize algorithms on same problem.
    
    Args:
        prices: Historical spot prices
        generation: Historical generation
        algorithms: List of algorithm names to test
        lambda_param: Risk aversion parameter
        beta: Fairness threshold
        period: Data period ('full', 'train', 'test') for file naming
        **kwargs: Additional arguments for optimise_ppa()
        
    Returns:
        (summary_df, results_dict) tuple containing:
            - DataFrame with comparison metrics
            - Dict mapping algorithm name to full OptimisationResult
            
    Example:
        >>> summary, results = compare_algorithms(
        ...     prices=prices,
        ...     generation=generation,
        ...     algorithms=['differential_evolution', 'dual_annealing'],
        ...     period='train'
        ... )
        >>> print(summary)
        >>> best_algo = summary.loc[summary['fitness'].idxmax(), 'algorithm']
    """
    results_dict = {}
    summary_rows = []
    
    print("=" * 80)
    print("ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"Testing algorithms: {algorithms}")
    print(f"Risk aversion (λ): {lambda_param}")
    print(f"Fairness constraint (β): {beta:.1%}")
    print(f"Period: {period}")
    print()
    
    for algo in algorithms:
        print(f"\n{'='*80}")
        print(f"ALGORITHM: {algo.upper()}")
        print(f"{'='*80}\n")
        
        try:
            result = optimise_ppa(
                prices=prices,
                generation=generation,
                lambda_param=lambda_param,
                beta=beta,
                algorithm=algo,
                period=period,
                verbose=True,
                **kwargs
            )
            
            results_dict[algo] = result
            
            summary_rows.append({
                'algorithm': algo,
                'floor': result.floor,
                'strike': result.strike,
                'cap': result.cap,
                'spread': result.spread,
                'fitness': result.fitness,
                'mean_revenue': result.mean_revenue,
                'cvar': result.cvar,
                'net_transfer_pct': result.net_transfer_pct,
                'time_sec': result.optimisation_time,
                'evaluations': result.num_evaluations,
                'is_feasible': result.is_feasible
            })
            
        except Exception as e:
            print(f"ERROR: {algo} failed with: {e}")
            continue
    
    summary_df = pd.DataFrame(summary_rows)
    
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 80)
    
    if len(summary_df) == 0:
        print("ERROR: All algorithms failed. Check error messages above.")
        return summary_df, results_dict
    
    print(summary_df[['algorithm', 'fitness', 'spread', 'time_sec', 'evaluations']].to_string(index=False))
    print()
    
    if len(summary_df) > 1:
        best_idx = summary_df['fitness'].idxmax()
        best_algo = summary_df.loc[best_idx, 'algorithm']
        best_fitness = summary_df.loc[best_idx, 'fitness']
        
        print(f"Best algorithm: {best_algo}")
        print(f"Best fitness: {best_fitness:,.0f} EUR")
        print()
    
    return summary_df, results_dict


def compare_risk_aversion(
    prices: pd.Series,
    generation: pd.Series,
    lambda_values: list = [0.0, 0.1, 0.3, 0.5, 0.7],
    beta: float = 0.05,
    **kwargs
) -> pd.DataFrame:
    """
    Optimise for multiple risk aversion levels to explore Pareto frontier.
    
    Args:
        prices: Historical spot prices
        generation: Historical generation
        lambda_values: List of λ values to test
        beta: Fairness threshold
        **kwargs: Additional arguments for optimise_ppa()
        
    Returns:
        DataFrame with results for each λ value
    """
    results = []
    
    print("=" * 80)
    print("RISK AVERSION SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Testing λ values: {lambda_values}")
    print(f"Fairness constraint: β={beta:.1%}")
    print()
    
    for lambda_param in lambda_values:
        print(f"\n{'='*80}")
        print(f"OPTIMISING FOR λ = {lambda_param}")
        print(f"{'='*80}\n")
        
        result = optimise_ppa(
            prices=prices,
            generation=generation,
            lambda_param=lambda_param,
            beta=beta,
            verbose=False,
            **kwargs
        )
        
        results.append({
            'lambda': lambda_param,
            'floor': result.floor,
            'strike': result.strike,
            'cap': result.cap,
            'spread': result.spread,
            'fitness': result.fitness,
            'mean_revenue': result.mean_revenue,
            'cvar': result.cvar,
            'net_transfer_pct': result.net_transfer_pct,
            'is_feasible': result.is_feasible
        })
        
        print(f"λ={lambda_param}: F={result.floor:.2f}, K={result.strike:.2f}, "
              f"C={result.cap:.2f}, Fitness={result.fitness:,.0f}")
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.to_string(index=False))
    print()
    
    return df
