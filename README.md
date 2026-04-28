# Asymptotic Analysis of Optimal Execution under Heston Stochastic Volatility: Extending the Almgren-Chriss Framework

## Abstract

The classic Almgren-Chriss (AC) model is a cornerstone of optimal execution, providing a closed-form solution for trading trajectories under the assumption of constant market volatility. However, empirical market data consistently exhibits stochastic volatility and mean-reversion, rendering the constant-volatility assumption insufficient for high-fidelity execution strategies. This project relaxes this assumption by introducing Heston dynamics into the market environment. We derive a modified trading strategy by solving the associated Hamilton-Jacobi-Bellman (HJB) equation via asymptotic perturbation methods, perturbing around the volatility-of-volatility parameter ($\xi$). We provide a first-order closed-form correction to the AC trajectory and validate its performance against the baseline model using a robust Monte Carlo simulation suite and rigorous statistical hypothesis testing.

## Mathematical Derivation Highlights
### 1. Stochastic Control Formulation
We define the value function $J(t,x,v)$ as the minimum expected implementation shortfall, where the state variables are time ($t$), trading rate ($x$), and volatility ($v$). Under Heston dynamics:

$dv_t=\theta(\omega-v_t)dt+\xi\sqrt{v_t}dW_t^{(v)}$

### 2. The HJB Equation
The resulting Hamilton-Jacobi-Bellman equation for this system is a second-order nonlinear PDE. Given the complexity of the Heston variance term, a direct global solution is often unavailable in simple closed form.

$\frac{\partial J}{\partial t}-\frac{\tau}{4\eta}\left(\frac{\partial J}{\partial x}\right)^2+\theta(\omega-v)\frac{\partial J}{\partial v}+\frac{1}{2}\xi^2v\frac{\partial^2J}{\partial v^2}+\rho\xi v \frac{\partial^2J}{\partial x\partial v}=0$.

### 3. Asymptotic Pertubation
By treating volatility-of-volatility ($\xi$) as a small pertubation parameter, we expand the value function: 

$J\approx J^{(0)}+\xi J^{(1)}+\mathcal{O}(\xi^2)$

Where $J^{(0)}$ recovers the original Almgren-Chriss solution and $J^{(1)}$ provides the first-order correction term that accounts for the volatility-of-volatility and the leverage effect ($\rho$).

## Key Features

- **Analytical Derivation**: Implementation of a first-order closed-form correction to the optimal trading rate.
- **Monte Carlo Validation**: A high-performance simulation engine that generates paired price paths (GBM vs. Heston) to isolate the impact of the stochastic volatility correction.
- **Statistical Rigor**: Evaluates "Superiority" through Paired T-Tests, Wilcoxon Signed-Rank tests for non-normal distributions, and Bootstrap-validated CVaR (Tail Risk) analysis.
- **Regime Sensitivity**: Automated analysis of how the model performs in high-initial-volatility vs. mean-reverting environments.

## Project Structure

├── core/
│   ├── AlmgrenChrissModel.py     # Base AC and Perturbed Heston logic
│   ├── MarketEnvironment.py      # SDE Integrators for GBM and Heston
│   └── MonteCarloSimulator.py    # Simulation engine
├── evaluation/
│   ├── comparator.py             # Orchestrates paired-path experiments
│   └── statistics.py             # Hypothesis testing and CVaR suite
├── main.py                       # Research interface and CLI
└── Research_Paper.pdf            # Full derivation and mathematical proof

## Implementation Note
This project was developed for MATH310: Introduction to Mathematical Modeling at the Colorado School of Mines. It represents a move away from heuristic-based trading toward a mathematically rigorous derivation of execution logic under stochastic regimes.
