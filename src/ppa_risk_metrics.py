"""
PPA risk metrics module for heuristic optimisation algorithms.

This module provides LEAN risk metrics specifically designed for optimisation.
Functions are optimised for performance as they are called thousands of times
during heuristic algorithm execution (GA/DE/PSO).

For comprehensive analysis and reporting, use ppa_analysis.py instead.

Key Metrics:
-----------
- VaR (Value at Risk): Quantile-based downside risk measure
- CVaR (Conditional VaR): Expected shortfall beyond VaR threshold
- Sharpe-like Ratio: Return-to-risk metric for balanced structuring

Design Philosophy:
-----------------
- Minimal return values: Only what optimisation algorithms need
- Fast execution: No unnecessary calculations
- Clear separation: Analysis/reporting belongs in ppa_analysis.py
"""

import pandas as pd
import numpy as np
from typing import Dict, Literal
from src.ppa_payoff import ppa_payoff


def calculate_var(
    revenues: pd.Series,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'YE'
) -> float:
    """
    Calculate Value at Risk (VaR) from revenue distribution.
    
    VaR represents the revenue threshold below which we expect to fall
    with probability (1 - confidence_level). For example, VaR at 95%
    confidence is the 5th percentile of the revenue distribution.
    
    Args:
        revenues: Hourly PPA revenues in EUR (indexed by datetime)
        confidence_level: Confidence level (default 0.95 for 95%)
        aggregation_freq: Frequency for aggregation before VaR calculation
                         ('YE' for annual, 'ME' for monthly, 'QE' for quarterly)
    
    Returns:
        VaR value in EUR at specified confidence level
        
    Example:
        var = calculate_var(hourly_revenues, confidence_level=0.95, aggregation_freq='YE')
        # Interpretation: "In 5% of years, annual revenue falls below EUR {var}"
    """
    if aggregation_freq:
        revenues = revenues.resample(aggregation_freq).sum()
        # Remove zero months (gaps in data, e.g., filtered years)
        revenues = revenues[revenues > 0]
    
    alpha = 1 - confidence_level
    var = revenues.quantile(alpha)
    
    return var


def calculate_cvar(
    revenues: pd.Series,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'YE'
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    
    CVaR is the expected revenue in the worst (1 - confidence_level) cases.
    It represents the average revenue when we're in the tail below VaR.
    
    Args:
        revenues: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh
        strike: Strike price in EUR/MWh
        cap: Cap price in EUR/MWh
        confidence_level: Confidence level for VaR/CVaR (default 0.95)
        aggregation_freq: Frequency for aggregation ('YE', 'ME', 'QE')
        risk_metric: Risk metric for Sharpe ratio ('std' or 'downside')
    
    Returns:
        CVaR value in EUR at specified confidence level
        
    Example:
        cvar = calculate_cvar(hourly_revenues, confidence_level=0.95, aggregation_freq='YE')
        # Interpretation: "In the worst 5% of years, average annual revenue is EUR {cvar}"
    """
    if aggregation_freq:
        revenues = revenues.resample(aggregation_freq).sum()
        # Remove zero months (gaps in data, e.g., filtered years)
        revenues = revenues[revenues > 0]
    
    alpha = 1 - confidence_level
    var = revenues.quantile(alpha)
    
    # CVaR is the mean of revenues below VaR threshold
    cvar = revenues[revenues <= var].mean()
    
    return cvar


def calculate_sharpe_ratio(
    revenues: pd.Series,
    aggregation_freq: str = 'YE',
    risk_metric: Literal['std', 'downside'] = 'std'
) -> float:
    """
    Calculate Sharpe-like ratio for PPA structure evaluation.
    
    Sharpe Ratio = Expected Return / Risk
    
    For PPA evaluation:
    - Return = Mean aggregated revenue
    - Risk = Standard deviation or downside deviation of aggregated revenue
    
    Args:
        revenues: Hourly PPA revenues in EUR (indexed by datetime)
        aggregation_freq: Frequency for aggregation ('YE', 'ME', 'QE')
        risk_metric: Risk measure ('std' for standard deviation,
                    'downside' for downside semi-deviation)
    
    Returns:
        Sharpe-like ratio (dimensionless)
        
    Note:
        Higher values indicate better risk-adjusted returns.
        This is a "Sharpe-like" ratio as we're not subtracting a risk-free rate.
    """
    aggregated = revenues.resample(aggregation_freq).sum()
    
    expected_return = aggregated.mean()
    
    if risk_metric == 'std':
        risk = aggregated.std()
    elif risk_metric == 'downside':
        # Downside deviation: only consider returns below mean
        mean = aggregated.mean()
        downside_returns = aggregated[aggregated < mean] - mean
        risk = np.sqrt((downside_returns ** 2).mean())
    else:
        raise ValueError(f"Invalid risk_metric '{risk_metric}'. Use 'std' or 'downside'.")
    
    if risk == 0:
        return np.inf if expected_return > 0 else 0.0
    
    sharpe_ratio = expected_return / risk
    
    return sharpe_ratio


def evaluate_for_optimisation(
    prices: pd.Series,
    generation: pd.Series,
    floor: float,
    strike: float,
    cap: float,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'YE',
    risk_metric: Literal['std', 'downside'] = 'std'
) -> Dict[str, float]:
    """
    Lean evaluation function for heuristic optimisation algorithms.
    
    Returns ONLY the metrics needed for:
    1. Objective function calculation (mean revenue, CVaR)
    2. Constraint validation (net transfer)
    
    Called thousands of times during optimisation - must be fast.
    For comprehensive analysis, use ppa_analysis.analyse_bilateral() instead.
    
    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh
        strike: Strike price in EUR/MWh
        cap: Cap price in EUR/MWh
        confidence_level: Confidence level for CVaR (default 0.95)
        aggregation_freq: Frequency for aggregation ('YE', 'ME', 'QE')
        risk_metric: Kept for backwards compatibility but not used in objective
    
    Returns:
        Dictionary containing ONLY:
        - 'mean_revenue': Mean aggregated revenue (EUR) [for objective]
        - 'cvar': Conditional Value at Risk (EUR) [for objective]
        - 'net_transfer': Offtaker net payment vs market (EUR) [for constraint]
        - 'market_baseline': Generator revenue at market prices (EUR) [for constraint normalisation]
        - 'var': Value at Risk (EUR) [diagnostic/reporting]
        - 'sharpe_ratio': Sharpe-like ratio [legacy, may be useful for comparison]
    
    Example:
        metrics = evaluate_for_optimisation(prices, gen, 40, 60, 80, lambda_param=0.5)
        
        # In GA/DE/PSO objective function:
        objective = metrics['mean_revenue'] - lambda_param * metrics['cvar']
        
        # In constraint checking:
        if metrics['net_transfer'] > beta * metrics['market_baseline']:
            return penalty  # Constraint violated
    """
    # Calculate PPA revenues (generator perspective)
    ppa_revenues = ppa_payoff(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        ppa_type='Collared'
    )
    
    # Market baseline (what generator would earn at spot prices)
    market_revenues = prices * generation
    
    # Aggregate for statistics
    aggregated_ppa = ppa_revenues.resample(aggregation_freq).sum()
    # Remove zero months (gaps in data, e.g., filtered years like 2022)
    aggregated_ppa = aggregated_ppa[aggregated_ppa > 0]
    
    # Core metrics for objective function: max E[R] - λ·CVaR
    mean_rev = aggregated_ppa.mean()
    cvar = calculate_cvar(ppa_revenues, confidence_level, aggregation_freq)
    
    # Constraint metric: net transfer for fairness constraint
    # Positive: off-taker pays more than market (subsidises generator)
    # Negative: off-taker pays less than market (benefits from PPA)
    net_transfer = (ppa_revenues - market_revenues).sum()
    
    # Market baseline for constraint normalisation
    total_market_revenue = market_revenues.sum()
    
    # Additional metrics (diagnostic/legacy)
    var = calculate_var(ppa_revenues, confidence_level, aggregation_freq)
    sharpe = calculate_sharpe_ratio(ppa_revenues, aggregation_freq, risk_metric)
    
    return {
        'mean_revenue': mean_rev,
        'cvar': cvar,
        'net_transfer': net_transfer,
        'market_baseline': total_market_revenue,
        'var': var,
        'sharpe_ratio': sharpe,
    }
