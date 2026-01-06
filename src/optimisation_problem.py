"""
Optimisation problem formulation for collared PPA structuring.

This module provides:
1. Constraint validation functions
2. Constraint handling mechanisms (penalty/rejection/repair)
3. Objective function wrapper for heuristic algorithms

Design: Hybrid constraint handling
- Structural constraints (F≤K≤C, bounds): Large penalty (effectively reject)
- Fairness constraint (net_transfer ≤ β): Linear penalty (soft constraint)

Usage:
    from src.optimisation_problem import objective_function
    
    # In GA/DE/PSO:
    fitness = objective_function(
        floor=40, strike=60, cap=80,
        prices=price_series, generation=gen_series,
        lambda_param=0.5, beta=0.05
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from src.ppa_risk_metrics import evaluate_for_optimisation


# ============================================================================
# CONFIGURATION: QUANTILE-BASED BOUNDS
# ============================================================================

def calculate_quantile_bounds(prices: pd.Series) -> Dict[str, Tuple[float, float]]:
    """
    Calculate quantile-based bounds for floor, strike, and cap parameters.
    
    Uses Spot market price distribution to set realistic search space bounds,
    avoiding extreme outliers from COVID-19 (2020) and Ukraine war (2022).
    
    Quantile ranges:
    - Floor: [max(0, Q₁%), Q₆₀%] - Lower tail to above median, never below 0
    - Strike: [Q₂₀%, Q₈₀%] - Broader range around median
    - Cap: [Q₄₀%, Q₉₉%] - From below median to high prices
    
    These ranges allow flexible collar structures:
    - Narrow collars: F≈K≈C around median
    - Wide collars: Low floor, high cap with strike in middle
    - Asymmetric collars: Various combinations
    
    Note: Floor lower bound is clamped to 0 because a PPA floor below zero
    would mean the generator pays to deliver power, which is economically
    nonsensical for a price protection mechanism.
    
    Args:
        prices: Historical Spot market prices (EUR/MWh)
    
    Returns:
        Dictionary with keys 'floor', 'strike', 'cap', each containing (min, max) tuple
        
    Example:
        bounds = calculate_quantile_bounds(prices)
        # {'floor': (0.0, 75.2), 'strike': (40.1, 95.8), 'cap': (58.7, 450.4)}
    """
    floor_min = max(0.0, prices.quantile(0.01))  # Clamp to 0
    
    return {
        'floor': (floor_min, prices.quantile(0.60)),
        'strike': (prices.quantile(0.20), prices.quantile(0.80)),
        'cap': (prices.quantile(0.40), prices.quantile(0.99)),
    }


# ============================================================================
# CONSTRAINT VALIDATION FUNCTIONS
# ============================================================================

def validate_structural_constraints(
    floor: float,
    strike: float,
    cap: float,
    bounds: Dict[str, Tuple[float, float]],
    min_spread: Optional[float] = None,
    check_quantile_bounds: bool = True
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate structural constraints for collared PPA parameters.
    
    Checks:
    1. F ≤ K ≤ C (ordering constraint) - ALWAYS enforced
    2. F_min ≤ F ≤ F_max (floor bounds) - Optional based on check_quantile_bounds
    3. K_min ≤ K ≤ K_max (strike bounds) - Optional based on check_quantile_bounds
    4. C_min ≤ C ≤ C_max (cap bounds) - Optional based on check_quantile_bounds
    5. C - F ≥ min_spread (optional minimum collar width)
    
    Args:
        floor: Floor price (EUR/MWh)
        strike: Strike price (EUR/MWh)
        cap: Cap price (EUR/MWh)
        bounds: Dictionary with 'floor', 'strike', 'cap' bounds
        min_spread: Optional minimum spread between cap and floor (EUR/MWh)
                   Note: Not currently used but available for future implementation.
                   If implemented, this would be another constraint subject to penalty.
        check_quantile_bounds: If True, enforce quantile bounds (for optimisation).
                               If False, only check F ≤ K ≤ C ordering (for stress testing).
                               Rationale: Contract signed in 2021 with F=39 EUR doesn't become
                               invalid because 2022 has different price quantiles.
    
    Returns:
        Tuple of (is_feasible, violations_dict)
        - is_feasible: True if all constraints satisfied
        - violations_dict: Magnitude of each violation (0 if satisfied)
    
    Example:
        # During optimisation: enforce all bounds
        feasible, violations = validate_structural_constraints(
            floor=40, strike=60, cap=80, bounds=bounds, check_quantile_bounds=True
        )
        
        # During stress testing: only check ordering
        feasible, violations = validate_structural_constraints(
            floor=40, strike=60, cap=80, bounds=bounds, check_quantile_bounds=False
        )
    """
    violations = {}
    
    # 1. Ordering constraint: F ≤ K ≤ C (ALWAYS enforced)
    violations['floor_strike'] = max(0, floor - strike)
    violations['strike_cap'] = max(0, strike - cap)
    
    # 2. Bounds constraints (only if check_quantile_bounds=True)
    if check_quantile_bounds:
        floor_min, floor_max = bounds['floor']
        strike_min, strike_max = bounds['strike']
        cap_min, cap_max = bounds['cap']
        
        violations['floor_lower'] = max(0, floor_min - floor)
        violations['floor_upper'] = max(0, floor - floor_max)
        violations['strike_lower'] = max(0, strike_min - strike)
        violations['strike_upper'] = max(0, strike - strike_max)
        violations['cap_lower'] = max(0, cap_min - cap)
        violations['cap_upper'] = max(0, cap - cap_max)
    else:
        # Stress testing mode: don't penalise quantile violations
        violations['floor_lower'] = 0
        violations['floor_upper'] = 0
        violations['strike_lower'] = 0
        violations['strike_upper'] = 0
        violations['cap_lower'] = 0
        violations['cap_upper'] = 0
    
    # 3. Minimum spread (optional, for future implementation)
    # if min_spread is not None:
    #     violations['min_spread'] = max(0, min_spread - (cap - floor))
    
    # Check if any violations
    total_violation = sum(violations.values())
    is_feasible = (total_violation == 0)
    
    return is_feasible, violations


def validate_fairness_constraint(
    net_transfer: float,
    market_baseline: float,
    beta: float = 0.05
) -> Tuple[bool, float]:
    """
    Validate fairness constraint: net_transfer / market_baseline ≤ β
    
    Ensures off-taker doesn't subsidise generator by more than β% (premium).
    
    Args:
        net_transfer: Total net transfer from off-taker to generator (EUR)
                     Σ(PPA_payments - market_payments) across all hours
        market_baseline: Total generator revenue at Spot market prices (EUR)
        beta: Maximum allowed net transfer as fraction of baseline (default 0.05 = 5%)
              Note: This is a fixed parameter (business decision), not optimised.
              For sensitivity analysis, run optimisation multiple times with different β values.
    
    Returns:
        Tuple of (is_feasible, violation_magnitude)
        - is_feasible: True if constraint satisfied
        - violation_magnitude: EUR-scaled violation (fraction × baseline)
                              This ensures penalties are meaningful relative to objective
    
    Example:
        feasible, violation = validate_fairness_constraint(
            net_transfer=50,000, market_baseline=1,000,000, beta=0.05
        )
        # net_transfer/baseline = 0.05 = 5%, exactly at threshold → feasible
        # violation = 0 EUR
        
        feasible, violation = validate_fairness_constraint(
            net_transfer=80,000, market_baseline=1,000,000, beta=0.05
        )
        # net_transfer/baseline = 0.08 = 8%, exceeds 5% by 3%
        # violation = 0.03 × 1,000,000 = 30,000 EUR
    """
    if market_baseline == 0:
        return False, float('inf')  # Invalid case
    
    normalized_transfer = net_transfer / market_baseline
    violation_fraction = max(0, normalized_transfer - beta)
    
    # Scale violation to EUR (percentage of baseline)
    violation_eur = violation_fraction * abs(market_baseline)
    
    is_feasible = (violation_fraction == 0)
    
    return is_feasible, violation_eur


def validate_all_constraints(
    floor: float,
    strike: float,
    cap: float,
    prices: pd.Series,
    generation: pd.Series,
    bounds: Dict[str, Tuple[float, float]],
    beta: float = 0.05,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'YE',
    check_quantile_bounds: bool = True
) -> Tuple[bool, Dict[str, float], Dict[str, float]]:
    """
    Comprehensive constraint validation: structural + fairness.
    
    Evaluates PPA structure and checks all constraints in one call.
    
    Args:
        floor: Floor price (EUR/MWh)
        strike: Strike price (EUR/MWh)
        cap: Cap price (EUR/MWh)
        prices: Historical Spot market prices (EUR/MWh)
        generation: Historical generation (MWh)
        bounds: Parameter bounds from calculate_quantile_bounds()
        beta: Fairness constraint threshold (default 0.05)
        confidence_level: For CVaR calculation (default 0.95)
        aggregation_freq: For revenue aggregation (default 'YE')
        check_quantile_bounds: If False, only enforce F ≤ K ≤ C (for stress testing)
    
    Returns:
        Tuple of (is_fully_feasible, structural_violations, fairness_violation_dict)
        
    Example:
        # During optimisation: check all constraints
        feasible, struct_viol, fair_viol = validate_all_constraints(
            floor=40, strike=60, cap=80,
            prices=prices, generation=gen,
            bounds=bounds, beta=0.05, check_quantile_bounds=True
        )
        
        # During stress testing: only check ordering
        feasible, struct_viol, fair_viol = validate_all_constraints(
            floor=40, strike=60, cap=80,
            prices=prices, generation=gen,
            bounds=bounds, beta=0.05, check_quantile_bounds=False
        )
    """
    # 1. Structural constraints (check first, fast fail)
    struct_feasible, struct_violations = validate_structural_constraints(
        floor, strike, cap, bounds, check_quantile_bounds=check_quantile_bounds
    )
    
    # 2. If structural constraints fail, skip fairness evaluation
    #    (ppa_payoff will raise error if F>K or K>C)
    if not struct_feasible:
        fairness_violation_dict = {
            'net_transfer': np.nan  # Cannot evaluate fairness with invalid structure
        }
        return False, struct_violations, fairness_violation_dict
    
    # 3. Evaluate metrics (only if structure is valid)
    metrics = evaluate_for_optimisation(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq
    )
    
    # 4. Fairness constraint
    fair_feasible, fair_violation = validate_fairness_constraint(
        net_transfer=metrics['net_transfer'],
        market_baseline=metrics['market_baseline'],
        beta=beta
    )
    
    # Overall feasibility
    is_fully_feasible = struct_feasible and fair_feasible
    
    fairness_violation_dict = {
        'net_transfer': fair_violation
    }
    
    return is_fully_feasible, struct_violations, fairness_violation_dict


# ============================================================================
# CONSTRAINT HANDLING: PENALTY FUNCTIONS
# ============================================================================

def calculate_penalty(
    structural_violations: Dict[str, float],
    fairness_violations: Dict[str, float],
    structural_penalty_weight: float = 1e8,
    fairness_penalty_weight: float = 10.0
) -> float:
    """
    Calculate penalty for constraint violations.
    
    Hybrid approach:
    - Structural violations: Very large fixed penalty (effectively reject)
    - Fairness violations: Moderate linear penalty (soft constraint)
    
    Args:
        structural_violations: Dict of structural constraint violations
        fairness_violations: Dict of fairness constraint violations
        structural_penalty_weight: Weight for structural violations (default 1e8)
                                  Very large value ensures solutions stay feasible
        fairness_penalty_weight: Weight for fairness violations (default 10)
                                Multiplies EUR-scaled violations for meaningful penalty
                                
                                Future options for fairness penalty:
                                - Quadratic: penalty = weight * violation²
                                - Adaptive: increase weight during optimisation
                                - Logarithmic: penalty = weight * log(1 + violation)
    
    Returns:
        Total penalty value to subtract from objective
        
    Example:
        penalty = calculate_penalty(struct_viol, fair_viol)
        fitness = base_objective - penalty
    """
    # Structural penalties (large, effectively rejection)
    structural_penalty = sum(structural_violations.values()) * structural_penalty_weight
    
    # Fairness penalties (linear, allows exploration)
    fairness_penalty = sum(fairness_violations.values()) * fairness_penalty_weight
    
    total_penalty = structural_penalty + fairness_penalty
    
    return total_penalty


# ============================================================================
# OBJECTIVE FUNCTION WRAPPER FOR HEURISTIC ALGORITHMS
# ============================================================================

def objective_function(
    floor: float,
    strike: float,
    cap: float,
    prices: pd.Series,
    generation: pd.Series,
    bounds: Dict[str, Tuple[float, float]],
    lambda_param: float = 0.5,
    beta: float = 0.05,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'ME',
    return_details: bool = False
) -> float:
    """
    Objective function for heuristic optimisation algorithms.
    
    Maximises: E[R_PPA] - λ·(E[R_PPA] - CVaR_q%(R_PPA))
             = (1-λ)·E[R_PPA] + λ·CVaR_q%(R_PPA)
    Subject to: Structural constraints + net_transfer/baseline ≤ β
    
    **With aggregation_freq='ME' (default):**
    - E[R_PPA] = Average monthly revenue (EUR per month)
    - CVaR = Average revenue in worst 5% of months (EUR per month)
    - Shortfall = E[R_PPA] - CVaR = Tail risk measure (how much worse bad months are)
    - λ weights the trade-off: higher λ → more focus on worst-case outcomes
    
    **Example:** Mean=302k, CVaR=36k, Shortfall=266k
    - λ=0: Fitness = 302k (only average matters)
    - λ=0.5: Fitness = 169k (balanced: 151k + 18k)
    - λ=1: Fitness = 36k (only worst case matters)
    
    Constraint handling: Hybrid penalty method
    - Structural violations → large penalty (effectively reject)
    - Fairness violations → linear penalty (soft constraint)
    
    Args:
        floor: Floor price (EUR/MWh)
        strike: Strike price (EUR/MWh)
        cap: Cap price (EUR/MWh)
        prices: Historical Spot market prices (EUR/MWh)
        generation: Historical generation (MWh)
        bounds: Parameter bounds from calculate_quantile_bounds()
        lambda_param: Risk aversion parameter (default 0.5)
                     λ=0: Risk-neutral (maximise mean only)
                     λ=1: Maximum risk aversion (minimise worst-case)
                     Higher λ → more conservative, focus on tail protection
        beta: Fairness constraint threshold (default 0.05 = 5%)
             Fixed parameter for sensitivity analysis
        confidence_level: For CVaR calculation (default 0.95)
        aggregation_freq: For revenue aggregation (default 'ME' = month-end)
                         Determines time period for mean and CVaR calculation
        return_details: If True, return (fitness, metrics, violations) tuple
    
    Returns:
        Fitness value (objective - penalties) for maximisation
        OR tuple of (fitness, metrics, violations) if return_details=True
        
    Example:
        # In GA/DE/PSO:
        fitness = objective_function(
            floor=40, strike=60, cap=80,
            prices=prices, generation=gen,
            bounds=bounds, lambda_param=0.5, beta=0.05
        )
        
        # For analysis:
        fitness, metrics, violations = objective_function(
            ..., return_details=True
        )
    """
    # 1. Validate structural constraints first (fast fail)
    struct_feasible, struct_viol = validate_structural_constraints(
        floor, strike, cap, bounds
    )
    
    # 2. If structural constraints fail, return penalised fitness immediately
    #    (cannot evaluate metrics with invalid structure)
    if not struct_feasible:
        penalty = calculate_penalty(
            structural_violations=struct_viol,
            fairness_violations={'net_transfer': 0.0}
        )
        fitness = -penalty  # Large negative value
        
        if return_details:
            violations = {
                'structural': struct_viol,
                'fairness': {'net_transfer': np.nan},
                'total_penalty': penalty,
                'is_feasible': False
            }
            return fitness, None, violations
        
        return fitness
    
    # 3. Calculate metrics (only if structure is valid)
    metrics = evaluate_for_optimisation(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq
    )
    
    # 4. Validate fairness constraint
    fair_feasible, fair_violation = validate_fairness_constraint(
        net_transfer=metrics['net_transfer'],
        market_baseline=metrics['market_baseline'],
        beta=beta
    )
    
    # 5. Base objective: max E[R] - λ·(E[R] - CVaR)
    #    Equivalently: (1-λ)·E[R] + λ·CVaR
    #    Shortfall measures tail risk: how much worse are bad months vs average
    shortfall = metrics['mean_revenue'] - metrics['cvar']
    base_objective = metrics['mean_revenue'] - lambda_param * shortfall
    
    # 6. Calculate penalty for fairness violations
    fair_viol = {'net_transfer': fair_violation}
    penalty = calculate_penalty(
        structural_violations={},  # Already validated, no structural violations
        fairness_violations=fair_viol
    )
    
    # 7. Final fitness (maximisation problem)
    fitness = base_objective - penalty
    
    # 8. Overall feasibility
    is_feasible = struct_feasible and fair_feasible
    
    if return_details:
        violations = {
            'structural': struct_viol,
            'fairness': fair_viol,
            'total_penalty': penalty,
            'is_feasible': is_feasible
        }
        return fitness, metrics, violations
    
    return fitness


# ============================================================================
# SENSITIVITY ANALYSIS HELPERS
# ============================================================================

def evaluate_multiple_beta(
    floor: float,
    strike: float,
    cap: float,
    prices: pd.Series,
    generation: pd.Series,
    bounds: Dict[str, Tuple[float, float]],
    beta_values: list = [0.03, 0.05, 0.07],
    lambda_param: float = 0.5
) -> pd.DataFrame:
    """
    Evaluate PPA structure across multiple β thresholds for sensitivity analysis.
    
    Helps answer: "How does solution quality change with fairness constraint strictness?"
    
    Args:
        floor, strike, cap: PPA parameters to evaluate
        prices, generation: Market data
        bounds: Parameter bounds
        beta_values: List of β values to test (default [0.03, 0.05, 0.07])
        lambda_param: Risk aversion parameter
    
    Returns:
        DataFrame with columns: beta, fitness, mean_revenue, cvar, net_transfer, is_feasible
        
    Example:
        results = evaluate_multiple_beta(40, 60, 80, prices, gen, bounds)
        print(results)
    """
    results = []
    
    for beta in beta_values:
        fitness, metrics, violations = objective_function(
            floor=floor,
            strike=strike,
            cap=cap,
            prices=prices,
            generation=generation,
            bounds=bounds,
            lambda_param=lambda_param,
            beta=beta,
            return_details=True
        )
        
        results.append({
            'beta': beta,
            'fitness': fitness,
            'mean_revenue': metrics['mean_revenue'],
            'cvar': metrics['cvar'],
            'net_transfer': metrics['net_transfer'],
            'net_transfer_pct': metrics['net_transfer'] / metrics['market_baseline'] * 100,
            'is_feasible': violations['is_feasible'],
            'penalty': violations['total_penalty']
        })
    
    return pd.DataFrame(results)
