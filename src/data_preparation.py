"""
Data preparation module for collared PPA optimisation.

This module loads, cleans, and processes raw price and generation data
into hourly granularity suitable for PPA analysis.
"""

import pandas as pd
from pathlib import Path
from typing import Literal, Optional


def data_preparation(
    data_dir: Path | str,
    technology: Literal['Solar', 'Wind Onshore'],
    country_code: str = 'DE',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    capacity_mw: float = 50.0,
    output_dir: Optional[Path | str] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare price and generation data for PPA analysis.

    This function:
    1. Loads all price and generation files from data directory
    2. Filters for Sequence 1 prices and specified technology
    3. Aggregates from 15-minute to hourly granularity
    4. Combines price and generation data into aligned dataframe
    5. Optionally saves processed data to CSV files

    Args:
        data_dir: Path to data directory containing raw CSV files
        technology: Generation technology type ('Solar', 'Wind Offshore', 'Wind Onshore')
        country_code: Two-letter country code for file prefix (e.g., 'DE', 'ES', 'UK')
        start_date: Start date in 'YYYY-MM-DD' format (None = use all available data)
        end_date: End date in 'YYYY-MM-DD' format (None = use all available data)
        capacity_mw: Synthetic asset capacity in MW for scaling generation profile
        output_dir: Optional path to save processed CSV files (None = don't save)

    Returns:
        Tuple of (price_df, generation_df, combined_df):
            - price_df: Hourly prices in EUR/MWh with datetime index
            - generation_df: Hourly generation in MWh with datetime index (scaled to capacity)
            - combined_df: Combined dataframe with both price and generation columns
    """
    data_dir = Path(data_dir)

    # Load raw data (scaling applied at file level in load function)
    price_data_raw = _data_preparation_load_prices(data_dir, country_code)
    generation_data_raw = _data_preparation_load_generation(
        data_dir, technology, capacity_mw, country_code)

    # Aggregate to hourly
    price_hourly = _data_preparation_aggregate_prices(price_data_raw)
    generation_hourly = _data_preparation_aggregate_generation(
        generation_data_raw)

    # Filter date range if specified
    if start_date or end_date:
        price_hourly = _data_preparation_filter_dates(
            price_hourly, start_date, end_date)
        generation_hourly = _data_preparation_filter_dates(
            generation_hourly, start_date, end_date)

    # Note: Scaling already applied at file level during load
    generation_scaled = generation_hourly

    # Combine price and generation data
    combined_df = _data_preparation_combine_data(
        price_hourly, generation_scaled, technology)

    # Save to CSV if output directory specified
    if output_dir is not None:
        _data_preparation_save_processed(
            price_hourly, generation_scaled, combined_df,
            Path(output_dir), technology, country_code
        )

    return price_hourly, generation_scaled, combined_df


def _data_preparation_load_prices(data_dir: Path, country_code: str) -> pd.DataFrame:
    """
    Load raw price data from all GUI_ENERGY_PRICES files.

    Filters for Sequence 1 only and extracts relevant columns.

    Args:
        data_dir: Path to data directory
        country_code: Two-letter country code for file prefix

    Returns:
        DataFrame with columns: timestamp_start, timestamp_end, price_eur_mwh
    """
    price_files = sorted(data_dir.glob(f'{country_code}_GUI_ENERGY_PRICES_*.csv'))

    if not price_files:
        raise FileNotFoundError(f"No price files found in {data_dir}")

    dfs = []
    for file in price_files:
        df = pd.read_csv(file)

        # Filter for Sequence 1 only (Germany) or Without Sequence (Spain/hourly data)
        # Germany has 15-min data with multiple sequences, Spain has hourly without sequences
        if 'Sequence Sequence 1' in df['Sequence'].values:
            df = df[df['Sequence'] == 'Sequence Sequence 1'].copy()
        elif 'Without Sequence' in df['Sequence'].values:
            df = df[df['Sequence'] == 'Without Sequence'].copy()
        else:
            # Fallback: take all data if sequence format unknown
            df = df.copy()

        # Extract start and end timestamps from MTU column
        df[['timestamp_start', 'timestamp_end']
           ] = df['MTU (CET/CEST)'].str.split(' - ', expand=True)
        # Remove timezone suffix if present (e.g., " (CET)" or " (CEST)")
        df['timestamp_start'] = df['timestamp_start'].str.replace(
            r' \(CE[S]?T\)', '', regex=True)
        df['timestamp_end'] = df['timestamp_end'].str.replace(
            r' \(CE[S]?T\)', '', regex=True)
        df['timestamp_start'] = pd.to_datetime(
            df['timestamp_start'], format='%d/%m/%Y %H:%M:%S')
        df['timestamp_end'] = pd.to_datetime(
            df['timestamp_end'], format='%d/%m/%Y %H:%M:%S')

        # Keep only relevant columns
        df = df[['timestamp_start', 'timestamp_end',
                 'Day-ahead Price (EUR/MWh)']].copy()
        df.rename(
            columns={'Day-ahead Price (EUR/MWh)': 'price_eur_mwh'}, inplace=True)

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    result.sort_values('timestamp_start', inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result


def _data_preparation_load_generation(
    data_dir: Path,
    technology: str,
    capacity_mw: float = 50.0,
    country_code: str = 'DE'
) -> pd.DataFrame:
    """
    Load raw generation data for specified technology.

    Filters for specified production type, scales to capacity, and extracts relevant columns.
    Scaling is applied at the file level to ensure consistent capacity across all time periods.

    Args:
        data_dir: Path to data directory
        technology: Production type to filter for
        capacity_mw: Target capacity in MW for scaling
        country_code: Two-letter country code for file prefix

    Returns:
        DataFrame with columns: timestamp_start, timestamp_end, generation_mw (scaled)
    """
    gen_files = sorted(data_dir.glob(
        f'{country_code}_AGGREGATED_GENERATION_PER_TYPE_GENERATION_*.csv'))

    if not gen_files:
        raise FileNotFoundError(f"No generation files found in {data_dir}")

    dfs = []
    for file in gen_files:
        df = pd.read_csv(file)

        # Filter for specified technology
        df = df[df['Production Type'] == technology].copy()

        # Extract start and end timestamps
        df[['timestamp_start', 'timestamp_end']
           ] = df['MTU (CET/CEST)'].str.split(' - ', expand=True)
        # Remove timezone suffix if present (e.g., " (CET)" or " (CEST)")
        df['timestamp_start'] = df['timestamp_start'].str.replace(
            r' \(CE[S]?T\)', '', regex=True)
        df['timestamp_end'] = df['timestamp_end'].str.replace(
            r' \(CE[S]?T\)', '', regex=True)
        df['timestamp_start'] = pd.to_datetime(
            df['timestamp_start'], format='%d/%m/%Y %H:%M:%S')
        df['timestamp_end'] = pd.to_datetime(
            df['timestamp_end'], format='%d/%m/%Y %H:%M:%S')

        # Handle 'n/e' (not available) values
        df['Generation (MW)'] = pd.to_numeric(
            df['Generation (MW)'], errors='coerce')

        # Scale to target capacity at file level
        current_max = df['Generation (MW)'].max()
        if current_max > 0 and not pd.isna(current_max):
            scaling_factor = capacity_mw / current_max
            df['Generation (MW)'] = df['Generation (MW)'] * scaling_factor

        # Keep only relevant columns
        df = df[['timestamp_start', 'timestamp_end', 'Generation (MW)']].copy()
        df.rename(columns={'Generation (MW)': 'generation_mw'}, inplace=True)

        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    result.sort_values('timestamp_start', inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result


def _data_preparation_aggregate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sub-hourly prices to hourly by averaging.

    Takes the mean of all price readings within each hour.

    Args:
        df: DataFrame with timestamp_start and price_eur_mwh columns

    Returns:
        DataFrame with datetime index (hour) and price_eur_mwh column
    """
    # Create hour column for grouping
    df['hour'] = df['timestamp_start'].dt.floor('h')

    # Average prices within each hour
    hourly = df.groupby('hour')['price_eur_mwh'].mean().reset_index()
    hourly.rename(columns={'hour': 'datetime'}, inplace=True)
    hourly.set_index('datetime', inplace=True)

    return hourly


def _data_preparation_aggregate_generation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 15-min generation (MW) to hourly (MWh) by averaging.

    Average MW over the hour gives MWh for that hour (power × time = energy).

    Args:
        df: DataFrame with timestamp_start and generation_mw columns

    Returns:
        DataFrame with datetime index (hour) and generation_mwh column
    """
    # Create hour column for grouping
    df['hour'] = df['timestamp_start'].dt.floor('h')

    # Average MW within each hour to get MWh
    # (Average power over 1 hour = energy in that hour)
    hourly = df.groupby('hour')['generation_mw'].mean().reset_index()
    hourly.rename(columns={'hour': 'datetime',
                  'generation_mw': 'generation_mwh'}, inplace=True)
    hourly.set_index('datetime', inplace=True)

    return hourly


def _data_preparation_filter_dates(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str]
) -> pd.DataFrame:
    """
    Filter dataframe to specified date range.

    Args:
        df: DataFrame with datetime index
        start_date: Start date in 'YYYY-MM-DD' format (None = no lower bound)
        end_date: End date in 'YYYY-MM-DD' format (None = no upper bound)

    Returns:
        Filtered DataFrame
    """
    result = df.copy()

    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        result = result[result.index >= start_dt]

    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        result = result[result.index <= end_dt]

    return result


def _data_preparation_scale_generation(
    df: pd.DataFrame,
    capacity_mw: float
) -> pd.DataFrame:
    """
    Scale generation profile to specified capacity.

    Scales the generation time series to represent an asset with
    the specified capacity in MW.

    Args:
        df: DataFrame with generation_mwh column
        capacity_mw: Target capacity in MW

    Returns:
        DataFrame with scaled generation_mwh values
    """
    result = df.copy()

    # Calculate current capacity (max generation in the time series)
    current_capacity = result['generation_mwh'].max()

    if current_capacity > 0:
        scaling_factor = capacity_mw / current_capacity
        result['generation_mwh'] = result['generation_mwh'] * scaling_factor

    return result


def _data_preparation_combine_data(
    price_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    technology: str
) -> pd.DataFrame:
    """
    Combine price and generation data into single aligned dataframe.

    Args:
        price_df: Hourly prices with datetime index
        generation_df: Hourly generation with datetime index
        technology: Technology name for column naming

    Returns:
        DataFrame with both price and generation columns, aligned by datetime
    """
    combined = pd.DataFrame(index=price_df.index)
    combined['price_eur_mwh'] = price_df['price_eur_mwh']
    combined[f'generation_mwh_{technology.lower().replace(" ", "_")}'] = generation_df['generation_mwh']

    # Remove rows where either value is missing
    combined.dropna(inplace=True)

    return combined


def _data_preparation_save_processed(
    price_df: pd.DataFrame,
    generation_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    output_dir: Path,
    technology: str,
    country_code: str
) -> None:
    """
    Save processed dataframes to CSV files.

    Args:
        price_df: Hourly prices dataframe
        generation_df: Hourly generation dataframe
        combined_df: Combined dataframe
        output_dir: Directory to save files
        technology: Technology name for file naming
        country_code: Two-letter country code for file prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tech_clean = technology.lower().replace(" ", "_")

    price_df.to_csv(output_dir / f'{country_code}_prices_hourly.csv')
    generation_df.to_csv(output_dir / f'{country_code}_generation_hourly_{tech_clean}.csv')
    combined_df.to_csv(output_dir / f'{country_code}_combined_hourly_{tech_clean}.csv')

    print(f"Processed {country_code} data saved to {output_dir}/")

def filter_by_period(
    prices: pd.Series,
    generation: pd.Series,
    period: Literal['full', 'train', 'test'] = 'full'
) -> tuple[pd.Series, pd.Series]:
    """
    Filter price and generation data by evaluation period.
    
    Periods:
    - 'full': 2015-2025 (all available data, 132 months)
    - 'train': 2015-2021 (84 months, CVaR uses 4.2 worst months)
    - 'test': 2023-2025 (36 months, CVaR uses 1.8 worst months)
    
    Note: Train/test split excludes 2022 (Ukraine war) which is treated as
    separate stress test. Train includes COVID recovery for diverse conditions.
    
    Args:
        prices: Hourly market prices (EUR/MWh) with datetime index
        generation: Hourly generation (MWh) with datetime index
        period: Period to filter ('full', 'train', 'test')
    
    Returns:
        Tuple of (filtered_prices, filtered_generation)
        
    Example:
        prices_train, gen_train = filter_by_period(prices, gen, 'train')
        # Returns 2015-2021 data only
    """
    if period == 'full':
        return prices, generation
    
    elif period == 'train':
        # 2015-01-01 to 2021-12-31 (inclusive)
        mask = (prices.index >= '2015-01-01') & (prices.index < '2022-01-01')
        
    elif period == 'test':
        # 2023-01-01 to 2025-12-31 (inclusive, excludes 2022 stress year)
        mask = (prices.index >= '2023-01-01') & (prices.index < '2026-01-01')
        
    else:
        raise ValueError(f"Invalid period '{period}'. Use 'full', 'train', or 'test'.")
    
    prices_filtered = prices[mask]
    generation_filtered = generation[mask]
    
    return prices_filtered, generation_filtered