"""
PPA payoff module for PPA optimisation.

This module implements revenue calculations for three PPA structures:
1. Fixed PPA: Single fixed price for all hours
2. Indexed PPA: Market price settlement (merchant exposure)
3. Collared PPA: Floor/Strike/Cap structure with risk-sharing

Mathematical Formulation:
------------------------

Fixed PPA:
    PPA(t) = Fixed_Price · G(t)

Indexed PPA:
    PPA(t) = P(t) · G(t)

Collared PPA:
           ⎧ F · G(t),    P(t) < F
    PPA(t) =   ⎨ K · G(t),    F ≤ P(t) < C
           ⎩ C · G(t),    P(t) ≥ C

Where:
- P(t) = Market price at time t (EUR/MWh)
- G(t) = Generation at time t (MWh)
- Fixed_Price = Fixed settlement price (EUR/MWh)
- F = Floor price (EUR/MWh) - minimum price seller receives
- K = Strike price (EUR/MWh) - "base" settlement price
- C = Cap price (EUR/MWh) - maximum price seller receives
- Constraint: F ≤ K ≤ C

Aggregation over period:
        T
V = Σ PPA(t)
       t=1

The value V represents the total PPA revenue over the period T (typically annual).

Collared PPA Interpretation (Seller's Perspective):
- When P(t) < F: Seller receives Floor (downside protection)
- When F ≤ P(t) < C: Seller receives Strike (normal settlement)
- When P(t) ≥ C: Seller receives Cap (upside limited)

The strike price is the "core" PPA price - the reference fixed price parties would have
agreed to under a standard fixed-price PPA. The floor and cap provide risk-sharing:
- Floor protects seller from low prices
- Cap limits seller's upside when prices are high (buyer benefit)
"""

import pandas as pd
from typing import Optional, Literal


def ppa_payoff(
    prices: pd.Series,
    generation: pd.Series,
    floor: Optional[float] = None,
    strike: Optional[float] = None,
    cap: Optional[float] = None,
    ppa_type: Literal['Fixed', 'Indexed', 'Collared'] = 'Collared',
    fixed_price: Optional[float] = None
) -> pd.Series:
    """
    Calculate hourly PPA revenue for any PPA structure type.

    Supports three PPA types:
    - Fixed: Single fixed price for all hours
    - Indexed: Market price settlement (merchant exposure)
    - Collared: Floor/Strike/Cap structure with risk-sharing

    Args:
        prices: Hourly market prices in EUR/MWh (indexed by datetime)
        generation: Hourly generation in MWh (indexed by datetime)
        floor: Floor price in EUR/MWh (required for Collared)
        strike: Strike price in EUR/MWh (required for Collared)
        cap: Cap price in EUR/MWh (required for Collared)
        ppa_type: Type of PPA ('Fixed', 'Indexed', or 'Collared')
        fixed_price: Fixed price in EUR/MWh (required for Fixed)

    Returns:
        Series of hourly PPA revenues in EUR (indexed by datetime)

    Raises:
        ValueError: If required parameters missing for specified PPA type
        ValueError: If constraints violated (e.g., floor > strike)
        ValueError: If prices and generation have different indices

    Examples:
        # Fixed PPA
        revenues = ppa_payoff(prices, gen, ppa_type='Fixed', fixed_price=55.0)

        # Indexed PPA (market exposure)
        revenues = ppa_payoff(prices, gen, ppa_type='Indexed')

        # Collared PPA
        revenues = ppa_payoff(prices, gen, floor=40, strike=60, cap=80, ppa_type='Collared')
    """
    # Validate indices match
    if not prices.index.equals(generation.index):
        raise ValueError(
            "Prices and generation must have identical datetime indices"
        )

    # Route to appropriate private function based on PPA type
    if ppa_type == 'Fixed':
        return _ppa_payoff_fixed(generation, fixed_price)
    elif ppa_type == 'Indexed':
        return _ppa_payoff_indexed(prices, generation)
    elif ppa_type == 'Collared':
        return _ppa_payoff_collared(prices, generation, floor, strike, cap)
    else:
        raise ValueError(
            f"Invalid ppa_type '{ppa_type}'. Must be 'Fixed', 'Indexed', or 'Collared'."
        )


def ppa_payoff_aggregate(
    hourly_payoffs: pd.Series,
    freq: str = 'YE'
) -> pd.Series:
    """
    Aggregate hourly PPA payoffs to specified frequency.

    Args:
        hourly_payoffs: Series of hourly payoffs in EUR
        freq: Aggregation frequency ('YE' for annual, 'M' for monthly, 'Q' for quarterly)

    Returns:
        Series of aggregated payoffs (V) by period
    """
    return hourly_payoffs.resample(freq).sum()


def ppa_payoff_total(
    prices: pd.Series,
    generation: pd.Series,
    floor: Optional[float] = None,
    strike: Optional[float] = None,
    cap: Optional[float] = None,
    ppa_type: Literal['Fixed', 'Indexed', 'Collared'] = 'Collared',
    fixed_price: Optional[float] = None
) -> float:
    """
    Calculate total PPA payoff (V) over entire period.

    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh (for Collared)
        strike: Strike price in EUR/MWh (for Collared)
        cap: Cap price in EUR/MWh (for Collared)
        ppa_type: Type of PPA ('Fixed', 'Indexed', or 'Collared')
        fixed_price: Fixed price in EUR/MWh (for Fixed)

    Returns:
        Total PPA value in EUR over the entire period
    """
    hourly = ppa_payoff(
        prices, generation, floor, strike, cap, ppa_type, fixed_price
    )
    return float(hourly.sum())


def ppa_payoff_statistics(
    prices: pd.Series,
    generation: pd.Series,
    floor: Optional[float] = None,
    strike: Optional[float] = None,
    cap: Optional[float] = None,
    ppa_type: Literal['Fixed', 'Indexed', 'Collared'] = 'Collared',
    fixed_price: Optional[float] = None
) -> dict[str, float]:
    """
    Calculate comprehensive statistics for PPA payoff.

    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh (for Collared)
        strike: Strike price in EUR/MWh (for Collared)
        cap: Cap price in EUR/MWh (for Collared)
        ppa_type: Type of PPA ('Fixed', 'Indexed', or 'Collared')
        fixed_price: Fixed price in EUR/MWh (for Fixed)

    Returns:
        Dictionary containing:
        - total_revenue: Sum of all hourly payoffs
        - mean_hourly_revenue: Mean hourly revenue in EUR
        - std_hourly_revenue: Standard deviation of hourly revenue in EUR
        - mean_hourly_price: Mean hourly payout price in EUR/MWh
        - std_hourly_price: Standard deviation of hourly payout price in EUR/MWh
        - min_hourly_revenue: Minimum hourly revenue in EUR
        - max_hourly_revenue: Maximum hourly revenue in EUR
        - hours_floor_active: Number of hours where floor was triggered (Collared only)
        - hours_cap_active: Number of hours where cap was triggered (Collared only)
        - hours_strike_active: Number of hours at strike price (Collared only)
    """
    hourly_revenue = ppa_payoff(
        prices, generation, floor, strike, cap, ppa_type, fixed_price
    )

    # Calculate payout price (revenue per MWh generated)
    # Filter out zero generation to avoid division by zero
    non_zero_gen = generation > 0
    hourly_price = pd.Series(index=hourly_revenue.index, dtype=float)
    hourly_price[non_zero_gen] = hourly_revenue[non_zero_gen] / \
        generation[non_zero_gen]
    hourly_price[~non_zero_gen] = 0.0

    stats = {
        'total_revenue': float(hourly_revenue.sum()),
        'mean_hourly_revenue': float(hourly_revenue.mean()),
        'std_hourly_revenue': float(hourly_revenue.std()),
        'mean_hourly_price': float(hourly_price[non_zero_gen].mean()),
        'std_hourly_price': float(hourly_price[non_zero_gen].std()),
        'min_hourly_revenue': float(hourly_revenue.min()),
        'max_hourly_revenue': float(hourly_revenue.max()),
    }

    # Add regime statistics for Collared PPA only
    if ppa_type == 'Collared':
        floor_active = (prices < floor).sum()
        cap_active = (prices >= cap).sum()
        strike_active = ((prices >= floor) & (prices < cap)).sum()

        stats.update({
            'hours_floor_active': int(floor_active),
            'hours_cap_active': int(cap_active),
            'hours_strike_active': int(strike_active)
        })

    return stats


def _ppa_payoff_fixed(
    generation: pd.Series,
    fixed_price: Optional[float]
) -> pd.Series:
    """
    Calculate hourly revenues for Fixed PPA.

    Args:
        generation: Hourly generation in MWh
        fixed_price: Fixed settlement price in EUR/MWh

    Returns:
        Series of hourly revenues in EUR

    Raises:
        ValueError: If fixed_price is None
    """
    if fixed_price is None:
        raise ValueError(
            "fixed_price must be provided for Fixed PPA type"
        )

    return fixed_price * generation


def _ppa_payoff_indexed(
    prices: pd.Series,
    generation: pd.Series
) -> pd.Series:
    """
    Calculate hourly revenues for Indexed PPA (market settlement).

    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh

    Returns:
        Series of hourly revenues in EUR
    """
    return prices * generation


def _ppa_payoff_collared(
    prices: pd.Series,
    generation: pd.Series,
    floor: Optional[float],
    strike: Optional[float],
    cap: Optional[float]
) -> pd.Series:
    """
    Calculate hourly revenues for Collared PPA.

    Args:
        prices: Hourly market prices in EUR/MWh
        generation: Hourly generation in MWh
        floor: Floor price in EUR/MWh
        strike: Strike price in EUR/MWh
        cap: Cap price in EUR/MWh

    Returns:
        Series of hourly revenues in EUR

    Raises:
        ValueError: If any of floor, strike, or cap is None
        ValueError: If floor > strike or strike > cap
    """
    if floor is None or strike is None or cap is None:
        raise ValueError(
            "floor, strike, and cap must all be provided for Collared PPA type"
        )

    _ppa_payoff_validate_collared_constraints(floor, strike, cap)

    # Calculate settlement prices based on market conditions
    settlement_prices = _ppa_payoff_get_settlement_prices(
        prices, floor, strike, cap
    )

    # Calculate hourly revenues: Settlement_Price × Generation
    return settlement_prices * generation


def _ppa_payoff_validate_collared_constraints(
    floor: float,
    strike: float,
    cap: float
) -> None:
    """
    Validate constraints for Collared PPA.

    Raises:
        ValueError: If constraints are violated
    """
    if floor > strike:
        raise ValueError(
            f"Floor ({floor}) must be ≤ Strike ({strike})"
        )

    if strike > cap:
        raise ValueError(
            f"Strike ({strike}) must be ≤ Cap ({cap})"
        )


def _ppa_payoff_get_settlement_prices(
    prices: pd.Series,
    floor: float,
    strike: float,
    cap: float
) -> pd.Series:
    """
    Determine settlement price for each hour based on market conditions.

    The settlement price is what the seller receives per MWh under the collared PPA:
    - Floor (F) if market price < floor (seller protected from low prices)
    - Strike (K) if floor ≤ market price < cap (normal PPA settlement)
    - Cap (C) if market price ≥ cap (seller's upside limited)

    Args:
        prices: Hourly market prices
        floor: Floor price
        strike: Strike price
        cap: Cap price

    Returns:
        Series of settlement prices (what seller receives per MWh)
    """
    settlement = pd.Series(index=prices.index, dtype=float)

    # Apply conditions
    settlement[prices < floor] = floor
    settlement[(prices >= floor) & (prices < cap)] = strike
    settlement[prices >= cap] = cap

    return settlement
