"""
PPA comprehensive analysis module for reporting and visualisation.

This module provides detailed analysis functions for research, reporting,
and visualisation. Functions return comprehensive breakdowns and are optimised
for insight generation rather than performance.

For optimisation algorithms, use ppa_risk_metrics.py instead (lean, fast functions).

Design Philosophy:
-----------------
- Comprehensive outputs: Everything you need for analysis
- Detailed breakdowns: Regime-by-regime, hour-by-hour statistics
- Reporting focus: Clear metrics for notebooks and presentations
"""

import pandas as pd
from typing import Dict, Tuple, Literal
from src.ppa_payoff import ppa_payoff, ppa_payoff_statistics
from src.ppa_risk_metrics import (
    calculate_var,
    calculate_cvar,
    calculate_sharpe_ratio,
    evaluate_for_optimisation
)


def analyse_generator_position(
    prices: pd.Series,
    generation: pd.Series,
    floor: float,
    strike: float,
    cap: float,
    aggregation_freq: str = 'YE'
) -> Dict[str, float]:
    """
    Detailed analysis of generator position relative to market.
    
    Breaks down generator gains and losses across all price regimes:
    - Downside protection: Extra revenue when PPA > market
    - Upside sacrifice: Lost revenue when PPA < market
    - Net benefit: Total gains - losses
    
    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh
        strike: Strike price in EUR/MWh
        cap: Cap price in EUR/MWh
        aggregation_freq: Frequency for aggregation ('YE', 'ME', 'QE')
    
    Returns:
        Dictionary containing:
        - 'ppa_revenue': Total PPA revenue (EUR)
        - 'market_revenue': Market baseline revenue (EUR)
        - 'gains_vs_market': Total gains when PPA > market (EUR)
        - 'losses_vs_market': Total losses when PPA < market (EUR)
        - 'net_benefit': Gains - losses = collar premium (EUR)
        - 'hours_gain': Count of hours with PPA > market
        - 'hours_loss': Count of hours with PPA < market
        - 'mean_agg_ppa': Mean aggregated PPA revenue (EUR)
        - 'mean_agg_market': Mean aggregated market revenue (EUR)
        - 'std_agg_ppa': Std dev of aggregated PPA revenue (EUR)
        - 'std_agg_market': Std dev of aggregated market revenue (EUR)
    """
    # Calculate revenues
    ppa_revenues = ppa_payoff(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        ppa_type='Collared'
    )
    market_revenues = prices * generation
    
    # Hourly differences: positive = gain, negative = loss
    hourly_diff = ppa_revenues - market_revenues
    
    # Gains and losses
    gains = hourly_diff[hourly_diff > 0].sum()
    losses = -hourly_diff[hourly_diff < 0].sum()  # Make positive
    
    # Hours count
    hours_gain = (hourly_diff > 0).sum()
    hours_loss = (hourly_diff < 0).sum()
    
    # Aggregated statistics
    agg_ppa = ppa_revenues.resample(aggregation_freq).sum()
    agg_market = market_revenues.resample(aggregation_freq).sum()
    
    return {
        'ppa_revenue': ppa_revenues.sum(),
        'market_revenue': market_revenues.sum(),
        'gains_vs_market': gains,
        'losses_vs_market': losses,
        'net_benefit': gains - losses,
        'hours_gain': int(hours_gain),
        'hours_loss': int(hours_loss),
        'mean_agg_ppa': agg_ppa.mean(),
        'mean_agg_market': agg_market.mean(),
        'std_agg_ppa': agg_ppa.std(),
        'std_agg_market': agg_market.std(),
    }


def analyse_offtaker_position(
    prices: pd.Series,
    generation: pd.Series,
    floor: float,
    strike: float,
    cap: float,
    aggregation_freq: str = 'YE'
) -> Dict[str, float]:
    """
    Detailed analysis of off-taker position relative to market.
    
    Correctly evaluates off-taker position across ALL regimes:
    - Floor regime: If floor > market → overpayment
    - Strike regime: If strike > market → overpayment; if strike < market → underpayment
    - Cap regime: If cap < market → underpayment (benefit)
    
    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh
        strike: Strike price in EUR/MWh
        cap: Cap price in EUR/MWh
        aggregation_freq: Frequency for aggregation ('YE', 'ME', 'QE')
    
    Returns:
        Dictionary containing:
        - 'ppa_cost': Total PPA cost (EUR)
        - 'market_cost': Market alternative cost (EUR)
        - 'overpayment': Total paid above market (EUR)
        - 'underpayment': Total saved below market (EUR)
        - 'net_transfer': Overpayment - underpayment (EUR)
        - 'premium_pct': (ppa_cost - market_cost) / market_cost × 100
        - 'hours_overpay': Count of hours where PPA > market
        - 'hours_underpay': Count of hours where PPA < market
        - 'mean_agg_cost': Mean aggregated PPA cost (EUR)
        - 'std_agg_cost': Std dev of aggregated PPA cost (EUR)
    
    Note:
        net_transfer > 0: Off-taker subsidises generator
        net_transfer < 0: Off-taker benefits from PPA
    """
    # Calculate payments
    ppa_payments = ppa_payoff(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        ppa_type='Collared'
    )
    market_payments = prices * generation
    
    # Hourly differences: positive = overpay, negative = underpay
    hourly_diff = ppa_payments - market_payments
    
    # Overpayment and underpayment
    overpayment = hourly_diff[hourly_diff > 0].sum()
    underpayment = -hourly_diff[hourly_diff < 0].sum()  # Make positive
    
    # Net transfer
    net_transfer = hourly_diff.sum()  # Same as overpayment - underpayment
    
    # Hours count
    hours_overpay = (hourly_diff > 0).sum()
    hours_underpay = (hourly_diff < 0).sum()
    
    # Aggregated statistics
    agg_ppa = ppa_payments.resample(aggregation_freq).sum()
    
    # Premium percentage
    total_ppa = ppa_payments.sum()
    total_market = market_payments.sum()
    premium_pct = ((total_ppa - total_market) / total_market * 100) if total_market != 0 else 0
    
    return {
        'ppa_cost': total_ppa,
        'market_cost': total_market,
        'overpayment': overpayment,
        'underpayment': underpayment,
        'net_transfer': net_transfer,
        'premium_pct': premium_pct,
        'hours_overpay': int(hours_overpay),
        'hours_underpay': int(hours_underpay),
        'mean_agg_cost': agg_ppa.mean(),
        'std_agg_cost': agg_ppa.std(),
    }


def analyse_bilateral(
    prices: pd.Series,
    generation: pd.Series,
    floor: float,
    strike: float,
    cap: float,
    confidence_level: float = 0.95,
    aggregation_freq: str = 'YE',
    risk_metric: Literal['std', 'downside'] = 'std'
) -> Dict[str, Dict]:
    """
    Comprehensive bilateral analysis of PPA structure.
    
    Combines generator and off-taker perspectives with risk metrics
    and fairness calculations. Returns everything needed for detailed
    analysis, reporting, and visualisation.
    
    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh
        strike: Strike price in EUR/MWh
        cap: Cap price in EUR/MWh
        confidence_level: Confidence level for VaR/CVaR
        aggregation_freq: Frequency for aggregation
        risk_metric: Risk metric for Sharpe ratio
    
    Returns:
        Dictionary with four sections:
        - 'optimisation': Core metrics from evaluate_for_optimisation()
        - 'generator': Detailed generator position analysis
        - 'offtaker': Detailed off-taker position analysis
        - 'regime_stats': Hours in each regime (floor/strike/cap)
        
    Example:
        analysis = analyse_bilateral(prices, gen, 40, 60, 80)
        
        print(f"Generator Sharpe: {analysis['optimisation']['sharpe_ratio']:.4f}")
        print(f"Collar Premium: {analysis['generator']['net_benefit']:,.0f} EUR")
        print(f"Offtaker Net Transfer: {analysis['offtaker']['net_transfer']:,.0f} EUR")
    """
    # Core optimisation metrics (lean)
    opt_metrics = evaluate_for_optimisation(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        confidence_level=confidence_level,
        aggregation_freq=aggregation_freq,
        risk_metric=risk_metric
    )
    
    # Detailed generator position
    gen_detail = analyse_generator_position(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        aggregation_freq=aggregation_freq
    )
    
    # Detailed off-taker position
    offtaker_detail = analyse_offtaker_position(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        aggregation_freq=aggregation_freq
    )
    
    # Regime statistics
    ppa_revenues = ppa_payoff(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        ppa_type='Collared'
    )
    regime_stats = ppa_payoff_statistics(prices, generation, floor, strike, cap)
    
    return {
        'optimisation': opt_metrics,
        'generator': gen_detail,
        'offtaker': offtaker_detail,
        'regime_stats': regime_stats,
    }


def compare_ppa_structures(
    prices: pd.Series,
    generation: pd.Series,
    ppa_configs: Dict[str, Tuple[float, float, float]],
    confidence_level: float = 0.95,
    aggregation_freq: str = 'YE'
) -> pd.DataFrame:
    """
    Compare multiple PPA parameter configurations side-by-side.
    
    Useful for analysing trade-offs between different structuring approaches
    and for presenting results from optimisation algorithms.
    
    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        ppa_configs: Dictionary mapping labels to (floor, strike, cap) tuples
                    Example: {'Conservative': (30, 50, 70),
                             'Aggressive': (50, 70, 90)}
        confidence_level: Confidence level for VaR/CVaR
        aggregation_freq: Frequency for aggregation
    
    Returns:
        DataFrame with configurations as rows and key metrics as columns
        
    Example:
        configs = {
            'Conservative': (30, 50, 70),
            'Moderate': (40, 60, 80),
            'Aggressive': (50, 70, 90)
        }
        comparison = compare_ppa_structures(prices, gen, configs)
        print(comparison)
    """
    results = []
    
    for label, (floor, strike, cap) in ppa_configs.items():
        analysis = analyse_bilateral(
            prices=prices,
            generation=generation,
            floor=floor,
            strike=strike,
            cap=cap,
            confidence_level=confidence_level,
            aggregation_freq=aggregation_freq
        )
        
        # Extract key metrics
        row = {
            'Structure': label,
            'Floor': floor,
            'Strike': strike,
            'Cap': cap,
            'Gen Sharpe': analysis['optimisation']['sharpe_ratio'],
            'Gen Mean Rev': analysis['optimisation']['mean_revenue'],
            'Gen VaR': analysis['optimisation']['var'],
            'Gen CVaR': analysis['optimisation']['cvar'],
            'Collar Premium': analysis['generator']['net_benefit'],
            'Offtaker Net Transfer': analysis['offtaker']['net_transfer'],
            'Market Baseline': analysis['optimisation']['market_baseline'],
            'Hours Floor': analysis['regime_stats']['hours_floor_active'],
            'Hours Strike': analysis['regime_stats']['hours_strike_active'],
            'Hours Cap': analysis['regime_stats']['hours_cap_active'],
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    return df


def rolling_period_analysis(
    prices: pd.Series,
    generation: pd.Series,
    floor: float,
    strike: float,
    cap: float,
    window_size: str = '365D',
    metric: Literal['sharpe_ratio', 'var', 'cvar', 'mean_revenue'] = 'sharpe_ratio',
    aggregation_freq: str = 'YE'
) -> pd.Series:
    """
    Test robustness of PPA parameters across rolling time windows.
    
    This function evaluates how stable the PPA structure performs across
    different market conditions by analysing rolling windows of data.
    
    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh
        strike: Strike price in EUR/MWh
        cap: Cap price in EUR/MWh
        window_size: Rolling window size (e.g., '365D' for 1 year)
        metric: Which metric to calculate
        aggregation_freq: Frequency for metric aggregation
    
    Returns:
        Series with metric values indexed by window end date
        
    Example:
        # Test how Sharpe ratio varies across different market years
        rolling_sharpe = rolling_period_analysis(
            prices, gen, 40, 60, 80,
            window_size='365D',
            metric='sharpe_ratio'
        )
        stability = rolling_sharpe.std()  # Lower = more stable
    """
    # Calculate hourly revenues once
    hourly_revenues = ppa_payoff(
        prices=prices,
        generation=generation,
        floor=floor,
        strike=strike,
        cap=cap,
        ppa_type='Collared'
    )
    
    # Create rolling window view
    rolling_obj = hourly_revenues.rolling(window=window_size, min_periods=1)
    
    results = []
    
    for window in rolling_obj:
        if len(window) < pd.Timedelta(window_size).total_seconds() / 3600:
            # Skip incomplete windows
            continue
            
        # Calculate requested metric for this window
        if metric == 'sharpe_ratio':
            value = calculate_sharpe_ratio(window, aggregation_freq)
        elif metric == 'var':
            value = calculate_var(window, aggregation_freq=aggregation_freq)
        elif metric == 'cvar':
            value = calculate_cvar(window, aggregation_freq=aggregation_freq)
        elif metric == 'mean_revenue':
            value = window.mean()
        else:
            raise ValueError(f"Invalid metric '{metric}'")
        
        results.append({'date': window.index[-1], 'value': value})
    
    return pd.DataFrame(results).set_index('date')['value']
