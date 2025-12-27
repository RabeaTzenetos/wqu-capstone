"""
Run data preparation for all renewable technologies.

Usage:
    python run_data_preparation.py           # Process all countries
    python run_data_preparation.py DE        # Process only Germany
    python run_data_preparation.py ES        # Process only Spain
"""

import sys
from pathlib import Path
from src.data_preparation import data_preparation

# Configuration
DATA_DIR = Path('data_raw')
OUTPUT_DIR = Path('data_processed')
TECHNOLOGIES = ['Solar', 'Wind Onshore']
CAPACITY_MW = 50.0  # Synthetic asset capacity
AVAILABLE_COUNTRIES = ['DE', 'ES'] 


def process_country(country_code: str):
    """Process data for a specific country."""
    print(f"\n{'=' * 60}")
    print(f"Processing {country_code} Market")
    print('=' * 60)

    for tech in TECHNOLOGIES:
        print(f"\nProcessing {country_code} - {tech}...")
        print("-" * 60)

        try:
            price_df, generation_df, combined_df = data_preparation(
                data_dir=DATA_DIR,
                technology=tech,
                country_code=country_code,
                capacity_mw=CAPACITY_MW,
                output_dir=OUTPUT_DIR
            )

            print(f"\n✓ {country_code} - {tech} processing complete!")
            print(f"  Price data: {len(price_df)} hourly records")
            print(f"  Generation data: {len(generation_df)} hourly records")
            print(f"  Combined data: {len(combined_df)} hourly records")
            print(
                f"  Date range: {combined_df.index.min()} to {combined_df.index.max()}")
            print(f"  Capacity scaled to: {CAPACITY_MW} MW")

            # Show statistics
            print(f"\n  Price Statistics (EUR/MWh):")
            print(f"    Min: {combined_df['price_eur_mwh'].min():.2f}")
            print(f"    Max: {combined_df['price_eur_mwh'].max():.2f}")
            print(f"    Mean: {combined_df['price_eur_mwh'].mean():.2f}")
            print(f"    Std: {combined_df['price_eur_mwh'].std():.2f}")

            gen_col = f'generation_mwh_{tech.lower().replace(" ", "_")}'
            print(f"\n  Generation Statistics (MWh):")
            print(f"    Min: {combined_df[gen_col].min():.2f}")
            print(f"    Max: {combined_df[gen_col].max():.2f}")
            print(f"    Mean: {combined_df[gen_col].mean():.2f}")
            print(f"    Std: {combined_df[gen_col].std():.2f}")

        except Exception as e:
            print(f"\n✗ Error processing {country_code} - {tech}: {e}")
            import traceback
            traceback.print_exc()


def main():
    print("=" * 60)
    print("Data Preparation for Collared PPA Optimisation")
    print("=" * 60)

    # Check if specific country requested via command line
    if len(sys.argv) > 1:
        country_code = sys.argv[1].upper()
        if country_code not in AVAILABLE_COUNTRIES:
            print(f"\n✗ Error: Country code '{country_code}' not recognized.")
            print(f"Available countries: {', '.join(AVAILABLE_COUNTRIES)}")
            sys.exit(1)
        
        countries_to_process = [country_code]
        print(f"\nProcessing single market: {country_code}")
    else:
        countries_to_process = AVAILABLE_COUNTRIES
        print(f"\nProcessing all markets: {', '.join(AVAILABLE_COUNTRIES)}")

    # Process each country
    for country in countries_to_process:
        process_country(country)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Processed files saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
