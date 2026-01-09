# Heuristic Optimisation for Pricing Collared PPA Structures

A comprehensive framework for pricing collared Power Purchase Agreement (PPA) structures using heuristic algorithms, bespoke objective functions as well as structural and fairness constraints. This project serves as a framework for evaluating how heuristic optimisation methods can determine optimal floor, strike, and cap levels that balance generator (seller) and off-taker (buyer) objectives and preferences.

## Research Question

**How can heuristic optimisation methods be used to determine optimal strike, floor, and cap levels in a collared PPA?**

## Overview

Collared PPAs are bilateral risk-sharing contracts between renewable energy generators and off-takers. This framework:

- **Implements** realistic collared PPA valuation with multiple structure types (Fixed, Indexed, Collared)
- **Optimises** PPA parameters using three heuristic algorithms from scipy
- **Evaluates** risk-adjusted returns using CVaR-based objective functions
- **Assesses** solution robustness using train/test splits and year-level market scenario analysis
- **Ensures** bilateral fairness through constraint enforcement

### Collared PPA Structure

For each hour, the generator receives:
- **Floor (F)** if spot price < floor → downside protection
- **Strike (K)** if floor ≤ spot price < cap → normal settlement
- **Cap (C)** if spot price ≥ cap → limited upside

**Constraint:** F ≤ K ≤ C

The optimisation balances generator risk-adjusted revenue against off-taker fairness constraints.

## Key Features

### PPA Types
- **Fixed PPA** - Single fixed price settlement
- **Indexed PPA** - Direct market (day-ahead spot) settlement
- **Collared PPA** - Floor/strike/cap structure with bilateral risk-sharing

### Algorithms Implemented
Three complementary scipy-based heuristic algorithms:

| Algorithm | Type | Best For |
|-----------|------|----------|
| **Differential Evolution (DE)** | Population-based evolutionary | General constrained optimisation, ~600 evaluations |
| **Dual Annealing (DA)** | Stochastic temperature-based | Non-linear/rugged landscapes, ~1,600 evaluations |
| **SHGO** | Deterministic simplicial | Provable convergence, ~45,000 evaluations |

### Objective Function
Risk-aware optimisation with bilateral constraints:

```
Maximise: 𝔼[R_PPA] - λ·(𝔼[R_PPA] - CVaR_q%(R_PPA))
        = (1-λ)·𝔼[R_PPA] + λ·CVaR_q%(R_PPA)

Where:
  - 𝔼[R_PPA] = Mean aggregated revenue (e.g., monthly average)
  - CVaR = Expected revenue in worst q% of periods (e.g., worst 5% of months)
  - λ ∈ [0,1] = Risk aversion parameter
    • λ=0: Risk-neutral (maximise mean)
    • λ=1: Maximum risk aversion (maximise worst-case)
  - Shortfall = 𝔼[R_PPA] - CVaR = Tail risk measure

Subject to:
  1. F ≤ K ≤ C (structural ordering constraint)
  2. Quantile-based bounds (avoids extreme outliers):
     - Floor: max(0, Q₁%) ≤ F ≤ Q₆₀% (never below 0)
     - Strike: Q₂₀% ≤ K ≤ Q₈₀%
     - Cap: Q₄₀% ≤ C ≤ Q₉₉%
  3. net_transfer / market_baseline ≤ β (fairness constraint)
```

### Risk Metrics
- **VaR (Value at Risk)** - Downside threshold (available for analysis)
- **CVaR (Conditional VaR)** - Tail risk quantification (used in objective function)
- **Sharpe Ratio** - Risk-adjusted returns (available for analysis)
- **Net Transfer** - Off-taker subsidy vs benefit (used in fairness constraint)

**Note:** All metrics are calculated during optimisation and available in results. The objective function currently uses mean revenue and CVaR, but can be customised (see Customisation section below).

## Data

**Market:** Spanish MIBEL (day-ahead spot prices)  
**Period:** 2015-2025 (11 years, 96,170 hours)  
**Source:** ENTSO-E Transparency Platform  
**Technology Focus:** Solar (Wind Onshore can also be processed and evaluated)
**Granularity:** Hourly & 15-min

**Data Splits:**
- **Full Period:** 2015-2025 (132 months) - algorithm comparison
- **Train:** 2015-2021 (84 months) - parameter optimisation
- **Test:** 2023-2025 (36 months) - out-of-sample evaluation
- **Scenario:** 2022-2025 as individual years - market scenario analysis

### Obtaining Data from ENTSO-E

Data must be downloaded from the **ENTSO-E Transparency Platform**: https://transparency.entsoe.eu/

**Steps to download:**

1. **Select market** on the map (here: Spain 'ES')

2. **For Day-Ahead Spot Prices:**
   - Navigate to: Market → Day-Ahead Prices
   - Spain 2015 direct link: https://transparency.entsoe.eu/market/energyPrices?permalink=694fdfc43b091c294b95e95b
   - Adjust timeframe (maximum 7 days but will download entire selected year)
   - Click three dots next to timeframe → Export to CSV (year) to download one file per year
   - Download one file per year (2015-2025)
   - Rename files to include 'ES' prefix: `ES_GUI_ENERGY_PRICES_YYYYMMDDHHSS-YYYYMMDDHHSS.csv`

3. **For Generation Data:**
   - Navigate to: Generation → Actual Generation per Production Type
   - Spain 2015 direct link: https://transparency.entsoe.eu/generation/actual/perType/generation?permalink=694fdf24e94a6e0252f95acc
   - Select technology: Solar, Wind Onshore, etc.
   - Adjust timeframe (maximum 7 days at a time but will download entire selected year)
   - Click three dots → Export to CSV (year) to download one file per year
   - Download one file per year (2015-2025)
   - Rename files to include 'ES' prefix: `ES_AGGREGATED_GENERATION_PER_TYPE_GENERATION_YYYYMMDDHHSS-YYYYMMDDHHSS.csv`

4. **Place downloaded files** in `data_raw/` directory

**Note:** If using markets other than DE (Germany) or ES (Spain), the data preparation pipeline may require adjustments for different data formats or granularities.

## Installation

### Prerequisites
- Python 3.11+
- Poetry (recommended) or pip for dependency management

### Setup

**Option 1: Using Poetry (Recommended for Development)**

```bash
# Clone the repository
git clone <repository-url>
cd wqu-capstone

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell

# Prepare data (processes raw ENTSO-E files to hourly granularity)
poetry run python run_data_preparation.py
```

**Option 2: Using pip**

```bash
# Clone the repository
git clone <repository-url>
cd wqu-capstone

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Prepare data
python run_data_preparation.py
```

## Usage

### 1. Data Preparation

Process raw ENTSO-E data files to hourly granularity:

**With Poetry:**
```bash
poetry run python run_data_preparation.py
```

**With pip:**
```bash
# Ensure virtual environment is activated
python run_data_preparation.py
```

Output: `data_processed/ES_*.csv` files with hourly prices and generation.

### 2. Run Optimisation

**With Poetry:**
```bash
# Full period optimisation (2015-2025)
poetry run python run_optimisation.py --period full

# Train period only (2015-2021)
poetry run python run_optimisation.py --period train

# Test period evaluation (2023-2025)
poetry run python run_optimisation.py --period test
```

**With pip:**
```bash
# Ensure virtual environment is activated

# Full period optimisation (2015-2025)
python run_optimisation.py --period full

# Train period only (2015-2021)
python run_optimisation.py --period train

# Test period evaluation (2023-2025)
python run_optimisation.py --period test
```

Results saved to:
- `results/tables/optimisation_<algorithm>_<period>.csv` - Best solutions
- `results/convergence/convergence_<algorithm>_<period>.json` - Convergence history
- `results/search_logs/search_log_<algorithm>_<period>.json` - All evaluations

### 3. Analysis Notebooks

Jupyter notebooks for exploration and evaluation:

```bash
# Launch Jupyter
poetry run jupyter notebook
```

**Available Notebooks:**
- `01_ES_data_exploration.ipynb` - Market and generation profile analysis
- `02_ES_ppa_modelling.ipynb` - PPA structure demonstration and sensitivity
- `04_ES_result_evaluation.ipynb` - Comprehensive algorithm evaluation (4 parts)

## Project Structure

```
wqu-capstone/
├── run_data_preparation.py          # Data processing pipeline
├── run_optimisation.py              # Main optimisation script
├── data_raw/                        # Raw ENTSO-E CSV files
├── data_processed/                  # Processed hourly data
├── src/
│   ├── data_preparation.py          # Data loading and filtering
│   ├── ppa_payoff.py                # PPA valuation functions
│   ├── ppa_risk_metrics.py          # Risk metrics (VaR, CVaR, Sharpe)
│   ├── ppa_analysis.py              # Bilateral analysis utilities
│   ├── optimisation_problem.py      # Objective & constraints
│   ├── ppa_optimiser.py             # Algorithm interface (DE/DA/SHGO)
│   ├── search_tracking.py           # Logging system
│   └── results_analysis.py          # Evaluation utilities
├── notebooks/                       # Jupyter analysis notebooks
├── tests/                           # Validation and baseline tests
└── results/                         # Outputs (tables, figures, logs)
```

## Results & Analysis

### Evaluation Framework

**Part 1: Algorithm Comparison (Full Period)**
- Convergence speed and efficiency
- Search space exploration patterns
- Solution quality and parameter values
- Computational cost analysis

**Part 2: Train/Test Sensitivity Analysis**
- In-sample vs out-of-sample performance
- Sensitivity of objective values across periods
- Constraint satisfaction robustness
- Algorithm comparison under distribution shift

**Part 3: Year-Level Market Scenario Analysis**
- Individual year performance (2022-2025)
- Extreme high price scenario testing (Russian invasion of Ukraine 2022)
- Wider price distribution and negative Spot price scenario testing (2024-2025)
- Collar activation distribution of algorithms across years


### Analysis Results

All analyses available in `notebooks/04_ES_result_evaluation.ipynb` with:
- Convergence plots (3-algorithm comparison)
- Train → test sensitivity analysis
- Market scenario analysis (year-by-year breakdown)

## Customisation

### Modifying the Objective Function

The objective function is defined in `src/optimisation_problem.py`:

```python
def objective_function(
    floor: float, strike: float, cap: float,
    prices: pd.Series, generation: pd.Series,
    lambda_param: float = 0.5,
    beta: float = 0.05
) -> float:
    """
    Objective: Maximise mean_revenue - lambda * (mean_revenue - CVaR)
             = (1-lambda) * mean_revenue + lambda * CVaR
    
    This balances mean returns vs worst-case tail outcomes.
    
    Modify this function to:
    - Change risk metric (e.g., use VaR instead of CVaR)
    - Add Sharpe ratio component
    - Adjust penalty weights for constraints
    - Implement multi-objective optimisation
    """
```

All risk metrics (VaR, CVaR, Sharpe, net transfer) are calculated by `evaluate_for_optimisation()` in `src/ppa_risk_metrics.py` and available for use.

### Modifying Constraints

Constraints are validated in `src/optimisation_problem.py`:

- **Structural constraints:** `validate_structural_constraints()` - F≤K≤C, quantile bounds
- **Fairness constraints:** `validate_fairness_constraint()` - net transfer threshold β
- **Bounds:** `calculate_quantile_bounds()` - modify quantile ranges (currently Q₁%, Q₆₀%, etc.)

Adjust penalty weights in `objective_function()` to change constraint enforcement strictness.

## License

MIT License

Copyright (c) 2025 Rabea Tzenetos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

Large portions of this codebase and documentation were generated and refactored using GitHub Copilot. All Copilot implementations were validated, tested, and refined by the author. The research design, theoretical framework, analysis, and interpretation remain entirely the work of the author.

## Citation

If you use this framework in your research, please cite:

Tzenetos, R. (2026). How can heuristic optimisation methods be used to determine 
optimal strike, floor, and cap levels in a collared PPA? MScFE Capstone Project, 
WorldQuant University.