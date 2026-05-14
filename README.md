# Asymptotic Analysis of Optimal Execution under Heston Stochastic Volatility: Extending the Almgren-Chriss Framework

## Abstract

The classic Almgren-Chriss (AC) model is a cornerstone of optimal execution, providing a closed-form solution for trading trajectories under the assumption of constant market volatility. However, empirical market data consistently exhibits stochastic volatility and mean-reversion, rendering the constant-volatility assumption insufficient for high-fidelity execution strategies. This project relaxes this assumption by introducing Heston dynamics into the market environment. We derive and implement a modified trading strategy by solving the associated Hamilton-Jacobi-Bellman (HJB) equation via asymptotic perturbation methods. The codebase provides a robust **Monte Carlo simulation suite** and **statistical evaluation framework** to validate the Heston-corrected strategy against the baseline AC model.

## Mathematical Context
### 1. Stochastic Control Formulation
We define the value function $J(t,x,v)$ as the minimum expected implementation shortfall, where the state variables are time ($t$), trading rate ($x$), and volatility ($v$). Under Heston dynamics:

$dv_t=\theta(\omega-v_t)dt+\xi\sqrt{v_t}dW_t^{(v)}$

### 2. The HJB Equation
The resulting Hamilton-Jacobi-Bellman equation for this system is a second-order nonlinear PDE:

$\frac{\partial J}{\partial t}-\frac{\tau}{4\eta}\left(\frac{\partial J}{\partial x}\right)^2+\theta(\omega-v)\frac{\partial J}{\partial v}+\frac{1}{2}\xi^2v\frac{\partial^2J}{\partial v^2}+\rho\xi v \frac{\partial^2J}{\partial x\partial v}=0$.

### 3. Asymptotic Pertubation
By treating volatility-of-volatility ($\xi$) as a small pertubation parameter, the value function is expanded: 

$J\approx J^{(0)}+\xi J^{(1)}+\mathcal{O}(\xi^2)$

Where $J^{(0)}$ recovers the original Almgren-Chriss solution and $J^{(1)}$ provides the first-order correction term that accounts for the volatility-of-volatility and the leverage effect ($\rho$). The resulting optimal trading rate implemented is:

$n_t^*\approx\kappa\coth(\kappa\tau)x_t-\frac{\xi\rho x_t}{4\eta}\tanh(\frac{\kappa\tau}{2})[\coth(\kappa\tau)-\kappa\tau\text{csch}^2(\kappa\tau)]$

## Key Features

- **Asymptotic Strategy Correction**: Implementation of a first-order closed-form correction to the AC trajectory that dynamically adjusts to volatility-of-volatility and the leverage effect.
- **LOBSTER Data Calibration**: A comprehensive `LobsterCalibrator` that estimates:
  - **Temporary impact (`eta`)** via book-walking simulations.
  - **Permanent impact (`gamma`)** via lagged signed order-flow regression.
  - **Heston parameters** (`v0`, `theta`, `omega`, `xi`, `rho`) from high-frequency mid-price returns and realized volatility series.
- **Monte Carlo Engine**: High-performance, multi-threaded simulation of paired price paths (Arithmetic Brownian Motion vs. Heston) to evaluate strategy performance under stochastic volatility.
- **Statistical Rigor**: A full suite of hypothesis tests comparing "Superiority" of strategies:
  - **Paired T-Tests** and **Wilcoxon Signed-Rank** tests for implementation shortfall (IS) means/medians.
  - **Levene's Test** for variance (risk) comparison.
  - **Bootstrap-validated CVaR** (Tail Risk) analysis at 95% and 99% levels.
  - **Regime Analysis**: Evaluating performance across low, mid, and high-volatility buckets.
- **Lambda Optimization**: Grid search and refined search tools to find the optimal risk-aversion parameter ($\lambda$) for a given environment.

## Project Structure
```
├── core/
│   ├── AlmgrenChrissModel.py     # AC solution with Heston asymptotic correction
│   ├── Backtester.py             # Single-path backtesting with optional correction flag
│   ├── MarketEnvironment.py      # SDE Integrators for ABM and Heston dynamics
│   └── MonteCarloSimulator.py    # Multi-threaded simulation engine with dynamic parameters
├── data/
│   ├── calibrator.py             # LOBSTER message/orderbook calibration logic
├── evaluation/
│   ├── comparator.py             # Orchestrates comparison between Classic AC and Heston-AC
│   └── statistics.py             # Comprehensive statistical test suite and plotting
├── main.py                       # CLI interface for calibration, simulation, and analysis
└── README.md                     # Project documentation
```

## Usage
The `main.py` script serves as the primary entry point, providing an interactive menu to:
1.  Calibrate parameters from LOBSTER datasets (CSV or ZIP format in the `data/` directory).
2.  Run single-path backtests to visualize trading trajectories (supports toggling the Heston correction).
3.  Execute Monte Carlo simulations for specific risk-aversion levels.
4.  Optimize $\lambda$ using a two-stage grid search.
5.  Run the full Model Comparison suite to evaluate the **Classic AC baseline** against the **Heston-Corrected strategy**.
6.  Perform a "Dataset Sweep" to aggregate results across multiple calibrated market samples.

## Known Issues and Rooms for Improvement
- **Feller Condition**: While the calibrator includes a "soft" Feller guard (adjusting $\xi$ if $2\theta\omega < \xi^2$), more robust handling of near-zero variance in the Euler-Maruyama scheme could be implemented.
- **Higher-Order Corrections**: The current implementation uses a first-order correction ($\mathcal{O}(\xi)$). Second-order terms ($\mathcal{O}(\xi^2)$) could further improve fidelity in high vol-of-vol environments.

## Implementation Note
This project was developed for MATH310: Introduction to Mathematical Modeling at the Colorado School of Mines. It represents a move away from heuristic-based trading toward a mathematically rigorous derivation of execution logic under stochastic regimes.
