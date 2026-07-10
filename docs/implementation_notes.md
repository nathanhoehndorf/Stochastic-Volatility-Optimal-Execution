# Implementation Notes: Optimal Trading Rate under Stochastic Volatility

This document provides an exhaustive explanation of every implementation decision in the codebase.
It covers mathematical foundations, discretization choices, engineering decisions, statistical methodology,
and all design tradeoffs discovered during development.

---

## Table of Contents

1. [Project Overview and Time-Scale Convention](#1-project-overview-and-time-scale-convention)
2. [core/AlmgrenChrissModel.py](#2-corealmgrenchrissmodelpy)
3. [core/MarketEnvironment.py](#3-coremarketenvironmentpy)
4. [core/Backtester.py](#4-corebacktesterpy)
5. [core/MonteCarloSimulator.py](#5-coremontecarlossimulatorpy)
6. [data/calibrator.py](#6-datacalibratorpy)
7. [evaluation/comparator.py](#7-evaluationcomparatorpy)
8. [evaluation/statistics.py](#8-evaluationstatisticspy)
9. [evaluation/correction_term_diagnostics.py](#9-evaluationcorrection_term_diagnosticspy)
10. [main.py](#10-mainpy)
11. [Cross-Cutting Engineering Decisions](#11-cross-cutting-engineering-decisions)

---

## 1. Project Overview and Time-Scale Convention

The project investigates whether an execution strategy that accounts for stochastic variance
(Heston dynamics) outperforms the classical constant-volatility Almgren-Chriss (AC) strategy.
The comparison is measured via *implementation shortfall* (IS): the gap between the ideal paper
value at time 0 and the cash actually realized by the time all shares are sold.

### T = 1 means one trading day

Every time-related parameter throughout the code is expressed in units of *one trading day*:

- `T = 1.0` is the default execution horizon, meaning the entire block is executed over one day.
- `sigma` is **daily** volatility, not annualized. The AC formula produces a kappa in
  units of day$^{-1/2}$, and dividing by $\sqrt{252}$ to obtain daily sigma from
  annualized sigma would be incorrect.
- Heston variance parameters `v0`, `omega` (long-run variance) are also in daily units.
  The calibrator explicitly multiplies local-interval variance by `intervals_per_day`
  before fitting Heston parameters to honor this convention.
- `N = 78` intervals corresponds to one 5-minute bucket per interval over a 6.5-hour day
  (390 trading minutes / 5 minutes = 78 intervals). `dt = T / N = 1/78` is therefore
  approximately 7.7 minutes expressed as a fraction of one day.

---

## 2. `core/AlmgrenChrissModel.py`

This is the mathematical core of the project.

### 2.1 Constructor parameters

```python
def __init__(self, X, T, N, sigma, lambd, eta, gamma,
             xi=None, rho=None, v0=None, theta=None, omega=None)
```

- **`X`**: Total number of shares to liquidate. This is a fixed quantity; the model decides
  *when* to sell, not *whether* to sell.
- **`T`**: Total trading horizon (days). 
- **`N`**: Number of uniform time intervals. The model assumes all intervals are equal-length
  (`dt = T/N`), which is a simplification but consistent with Almgren-Chriss (2001).
- **`sigma`**: Asset volatility. In the classic AC model this is the only volatility input.
  When Heston parameters are also provided, `sigma` controls the base AC schedule
  (which uses `sigma^2` as the constant variance proxy). When `v0` is not provided,
  the code defaults `self.v0 = sigma**2` so the correction term's deterministic
  variance fallback is consistent with the AC baseline.
- **`lambd`** (written as `lambd` to avoid Python's `lambda` keyword): The risk-aversion
  coefficient. Higher `lambd` makes the trader more variance-averse and front-loads the
  schedule. This is the single tunable parameter when running the lambda optimization grid search.
- **`eta`**: The *temporary market impact* coefficient. Each trade of `n` shares over interval
  `dt` moves the execution price by `eta * (n/dt)` below the unaffected price. This is a
  linear model for temporary impact; the real price-impact function may be concave or convex,
  but linear is the standard AC assumption.
- **`gamma`**: The *permanent market impact* coefficient. Every share sold permanently depresses
  the mid-price by `gamma` per share. This is accumulated across the entire schedule.
- **`xi`** (vol of vol): The volatility of the variance process in the Heston model. When
  `None`, the model falls back to classic AC behavior throughout.
- **`rho`**: The correlation between Brownian motions driving price and variance.
  Negative `rho` is the *leverage effect*: when price falls, variance tends to rise.
  This is empirically observed in equities and is critical for the correction term
  to produce a non-trivial signal.
- **`v0`**: Initial instantaneous variance. Defaults to `sigma**2` if not provided.
- **`theta`**: Mean-reversion speed of variance. In Heston notation, the SDE for variance is
  `dv = theta*(omega - v) dt + xi*sqrt(v) dW`. Larger `theta` means variance snaps back
  to `omega` faster.
- **`omega`**: Long-run variance. Variance is attracted to `omega` on a timescale of
  `1/theta` (in the same time units as `T`). Defaults to `v0` if not provided, which
  means no mean-reversion drift in the deterministic baseline.

```python
self.dt = T / N
self.times = np.linspace(0, T, N + 1)
```

`times` is a grid of `N+1` equally-spaced time points from 0 to T inclusive,
representing the *start* of each interval plus the terminal point.

---

### 2.2 `compute_kappa()`

$$\kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}$$

Kappa is the fundamental decay rate of the AC optimal inventory trajectory. It controls how
"urgently" the trader sells:
- Large `kappa` → fast decay → front-loaded schedule (aggressive)
- Small `kappa` → slow decay → nearly uniform schedule (passive)

The formula comes from solving the HJB equation for the AC optimization problem.
`sigma^2` in the numerator is the per-day variance (not annualized).

---

### 2.3 `compute_inventory_trajectory()` — Classic AC Solution

```python
res[nonzero] = self.X * np.sinh(kappa * tau[nonzero]) / np.sinh(kappa * T)
```

The closed-form optimal inventory trajectory from Almgren and Chriss (2001) is:

$$x_t = X \cdot \frac{\sinh(\kappa(T-t))}{\sinh(\kappa T)}$$

where `tau = T - t` is the remaining time. This is the **exact** analytical solution and is
used as the baseline schedule everywhere in the code.

**Numerical overflow guard**: When `kappa * T > 700`, the direct `sinh` calculation overflows
IEEE 754 double precision. The code detects this and rewrites the formula using logarithms:

```python
log_ratio = -kappa * (T - tau)
adjustment = (1 - exp(-2*kappa*tau)) / (1 - exp(-2*kappa*T))
return X * exp(log_ratio) * adjustment
```

This is mathematically equivalent but avoids overflow. The threshold 700 is chosen because
`exp(700) ≈ 10^{304}`, already near double-precision limits.

The final time point `tau = 0` is handled explicitly by setting `res[~nonzero] = 0`,
because `sinh(0)/sinh(kappa*T)` would be exact zero but might fail the mask if `kappa*T`
itself is degenerate.

---

### 2.4 `deterministic_variance(t)`

$$v_t \approx \omega + (v_0 - \omega) e^{-\theta t}$$

This is the *mean* of the Heston variance process conditioned on `v0` at time 0.
Under the Heston SDE, the expected value of `v_t` is exactly this expression.
It is used as a fallback when no realized variance path is available (e.g., for
the static precomputed correction in `compute_b_trajectory()`).

If `theta` is `None`, the function returns `v0` everywhere, which is equivalent to
treating variance as a constant equal to its initial value.

---

### 2.5 `compute_b_value(t)` — Quadrature for the HJB Correction Coefficient

$$b(t) = \frac{\lambda}{\xi} \int_t^T e^{-\theta(s-t)} \left(\frac{\sinh(\kappa(T-s))}{\sinh(\kappa(T-t))}\right)^2 ds$$

`b(t)` arises from the first-order perturbation expansion of the value function in the
HJB problem under Heston dynamics. Specifically, the value function is expanded as
$V \approx V_0 + \xi V_1$, where $V_0$ is the classic AC value function. The cross-derivative
$\partial^2 V_1 / \partial x \partial v$ is proportional to $b(t)$.

**Why quadrature, not closed form**: The integrand involves both the mean-reversion
exponential and the sinh-ratio squared from the AC trajectory. A closed-form antiderivative
exists in some limits but is unwieldy for general `theta`. `scipy.integrate.quad` uses
adaptive Gaussian quadrature with error control, which is accurate and fast for smooth
one-dimensional integrands.

**Guard conditions**:
- `tau <= 1e-10`: At the terminal time, `b(T) = 0` by construction (no remaining integration range).
- `xi is None` or `abs(xi) <= 1e-12`: If there is no vol-of-vol, there is no correction.
- `theta is None`: Cannot evaluate the exponential decay without `theta`.
- `abs(kappa) <= 1e-12`: Degenerate risk-neutral case; skip.
- `abs(sinh(kappa*tau)) <= 1e-12`: Would cause division by zero in the ratio.

**Small-kappa fallback inside the integrand**:

```python
if kappa < 1e-4:
    ratio = (self.T - s) / tau
```

When `kappa` is very small, `sinh(kappa*x) ≈ kappa*x` and the ratio simplifies to
`(T-s)/(T-t)`. This avoids catastrophic cancellation in the sinh ratio at near-zero kappa.

The prefactor `lambd / xi` carries units: `lambda` has units of `1/(shares^2 * variance * time)`
from the AC objective, and `xi` has units of `1/sqrt(time)` in variance dynamics. The product
makes `b(t)` dimensionally correct for the correction term.

---

### 2.6 `compute_b_trajectory()`

Evaluates `compute_b_value(t)` at each of the `N+1` grid points. This is called once per
path during `compute_trade_list`, so the quadrature runs `N+1` times per trade-list computation.
For `N=78`, this means 79 adaptive quadrature calls, which is fast in practice but is worth
caching if performance becomes critical.

---

### 2.7 `_resolve_variance_path(variance_path=None)`

Centralizes the logic for deciding which variance values to use in the correction:

- If a realized `variance_path` (length `N+1`) is provided, validate its length and use it directly.
- If `None`, compute the deterministic mean path using `deterministic_variance(t)` at each grid point.

The length check (`variance_path.shape[0] != self.N + 1`) enforces that the variance path and
the time grid are aligned: each variance value `v_k` corresponds to the *start* of interval `k`.

---

### 2.8 `_compute_trade_correction_chunks(variance_path)`

This method computes the raw, per-step correction to the trading rate:

$$\text{corr\_rate}_k = \frac{\xi \rho \, v_k \cdot (\xi b_k)}{2 \eta \kappa} \tanh\!\left(\frac{\kappa \tau_k}{2}\right)$$

Then multiplies by `dt` to produce a correction in shares (not shares/time):

```python
correction_chunks[k] = corr_rate * self.dt
```

**Sign convention**: The correction rate is subtracted from the AC rate in the
continuous-time formula, but in discrete time the correction chunk is *added* to the
AC trade quantity. The sign of `rho` carries the directional effect: for negative `rho`
(leverage effect) and positive variance, the correction is negative, meaning the strategy
slows down trading during high-variance periods.

**`h_v = xi * b_values[k]`**: This is $h_v = \partial h / \partial v$, the derivative of
the value function's first-order Heston correction with respect to variance. From the
HJB expansion, this equals $\xi \cdot b(t)$.

**Why `tanh(kappa * tau / 2)`**: This factor comes from the integral of the AC trajectory
shape over the remaining horizon. It equals zero at `tau = 0` (no more trading) and
approaches 1 for large `kappa * tau` (lots of time and urgency remaining).

**Degenerate guards**: If `kappa < 1e-12` or `tau < 1e-10`, the correction is set to zero.
The `max(variance_path[k], 0.0)` ensures the variance used is non-negative (truncation).

---

### 2.9 `compute_perturbed_inventory_trajectory(variance_path=None)`

Computes the inventory path that corresponds to the corrected trade schedule. This is a
*derived* quantity: it calls `compute_trade_list(use_correction=True)` and then reconstructs
inventory by cumulative subtraction:

```python
x[1:] = self.X - np.cumsum(trades)
x = np.maximum(x, 0.0)
x[-1] = 0.0
```

The `maximum(x, 0.0)` prevents inventory from going negative due to floating-point errors.
The explicit `x[-1] = 0.0` enforces full liquidation at terminal time.

**Design note**: This function previously integrated the AC ODE with Euler steps, which
introduced a systematic discretization mismatch between the AC baseline (exact analytic)
and the corrected path. The refactoring to derive the perturbed inventory *from* the
corrected trade list eliminates that bias.

---

### 2.10 `compute_trade_list(use_correction=False, variance_path=None)`

This is the primary public method for obtaining a trading schedule.

**Classic AC path (`use_correction=False`)**:
```python
x_classic = self.compute_inventory_trajectory()   # exact analytic
classic_trades = x_classic[:-1] - x_classic[1:]  # differences give shares per interval
return classic_trades
```

The trade at step `k` is `x_k - x_{k+1}`, i.e. the decrease in inventory.
Using the exact analytic inventory ensures the classic schedule is free from
discretization error.

**Heston-corrected path (`use_correction=True`)**:

```python
variance_values = self._resolve_variance_path(variance_path)
correction_chunks = self._compute_trade_correction_chunks(variance_values)

corrected_trades = np.zeros(self.N, dtype=float)
current_x = float(self.X)

for k in range(self.N):
    if current_x <= 1e-12:
        corrected_trades[k] = 0.0
        continue

    discrete_correction = correction_chunks[k] * current_x
    proposed_trade = classic_trades[k] + discrete_correction

    proposed_trade = min(max(proposed_trade, 0.0), current_x)
    corrected_trades[k] = proposed_trade
    current_x -= proposed_trade

corrected_trades *= self.X / total_sold
```

**Why multiply `correction_chunks[k]` by `current_x`**:

The raw correction chunk from `_compute_trade_correction_chunks` is dimensionally a
correction to the *fractional* trading rate (a normalized velocity adjustment).
From the HJB expansion, the first-order correction to the optimal velocity is linear in
remaining inventory $x_t$. Without multiplying by $x_t$, the correction represents an
adjustment for a portfolio of one share, making it irrelevant for portfolios of thousands
of shares.

This was verified empirically: with `current_x` scaling, `delta_trade` reached 0.8–3.2 shares
per step, while without it the maximum was 0.0003 shares — economically invisible.

**`current_x` tracking**: Rather than using the AC inventory values `ac_x[:-1]`, the code
tracks *corrected* inventory step by step. This ensures that if an early correction causes
slightly more or less selling, the available inventory for subsequent steps is updated
consistently.

**Feasibility clamp**:
```python
proposed_trade = min(max(proposed_trade, 0.0), current_x)
```
- Lower bound 0: No buying back (short-selling not allowed in this model).
- Upper bound `current_x`: Cannot sell more shares than currently held.

**Renormalization**:
```python
corrected_trades *= self.X / total_sold
```
Because the step-by-step feasibility clamp may cause the total sold to deviate slightly
from `X`, all trades are scaled proportionally to guarantee exact liquidation.
This preserves the *shape* of the correction while restoring the inventory constraint.
The `if total_sold <= 1e-12` guard handles pathological degenerate cases where all
corrected trades collapsed to zero.

---

### 2.11 `summary(use_correction=False, variance_path=None)`

Returns a dictionary containing all major outputs: kappa, time grid, inventory path,
trade list, `b` values (if correction used), dt, and a flag indicating whether the
perturbed path was used. This is primarily useful for inspection and debugging,
not for the main Monte Carlo loops.

---

## 3. `core/MarketEnvironment.py`

The `MarketEnvironment` holds all market parameters and provides price-path simulation
and implementation-shortfall computation. It is designed to be stateless with respect to
randomness — each simulation call produces an independent path from a given seed.

### 3.1 Constructor

```python
def __init__(self, S0, sigma, T, N, gamma, eta, heston_params=None)
```

- **`S0`**: Initial (unaffected) mid-price. The implementation shortfall uses `X * S0`
  as the paper value against which realized cash is compared.
- **`sigma`**: Constant volatility, used for ABM and GBM simulation alternatives.
  Not used inside the Heston simulator.
- **`gamma`**, **`eta`**: Market impact coefficients, mirrored from `AlmgrenChrissModel`
  to keep market mechanics self-contained.
- **`heston_params`**: Dictionary with keys `v0, mu, theta, omega, xi, rho`. Stored on the
  environment so the `Backtester` can simulate Heston paths using calibrated parameters
  without those parameters being hardcoded.

```python
self.dt = T / N
self.sqrt_dt = np.sqrt(self.dt)   # pre-computed for ABM performance
```

---

### 3.2 `simulate_heston_paths_vectorized(...)`

Generates `n_sims` Heston price and variance paths simultaneously using NumPy broadcasting.
This is the batch version for situations where many paths are needed at once (e.g., the
vectorized Backtester use case or potential future parallel IS estimation).

**Cholesky correlation**:
```python
corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
L = np.linalg.cholesky(corr_matrix)
z = rng.standard_normal((n_sims, N, 2))    # independent
z_corr = z @ L.T                            # correlated
Z1 = z_corr[:, :, 0]  # variance shocks
Z2 = z_corr[:, :, 1]  # price shocks
```

Two independent standard normals per step per path are drawn. Multiplying by the lower
Cholesky factor `L` produces correlated pairs. This is the standard and numerically stable
way to generate bivariate normal samples with a given correlation.

**Log-Euler discretization for price**:
```python
S[:, k+1] = S[:, k] * np.exp(
    (mu - 0.5 * v_safe) * self.dt
    + np.sqrt(v_safe * self.dt) * Z2[:, k]
)
```

Using the log-Euler scheme (Euler applied to `log S`) avoids the possibility of prices
going negative, which the plain Euler scheme can produce. The Ito correction term
`-0.5 * v_safe` is the standard log-return drift adjustment.

**Full truncation for variance**:
```python
v_safe = np.maximum(v[:, k], 0.0)
v[:, k+1] = v[:, k] + theta * (omega - v_safe) * dt + xi * sqrt(v_safe * dt) * Z1
v[:, k+1] = np.maximum(v[:, k+1], 0.0)
```

The variance process `v_t` can become negative in naive Euler discretization when `xi` is
large or `dt` is large. The "full truncation" scheme addresses this by:
1. Clamping `v_safe = max(v_k, 0)` before it appears in any square root or drift.
2. Flooring `v_{k+1}` at zero after the update.

This is the most common robust discretization for CIR/Heston variance processes.

---

### 3.3 `simulate_unaffected_price_heston(...)`

Single-path version of the Heston simulator. Generates one price path and one variance path
using the same log-Euler full-truncation scheme as the vectorized version, but in a scalar loop.
This is used throughout the comparator and backtester where one path is needed at a time.

The `rng = np.random.default_rng(seed)` uses NumPy's modern Generator API. Passing an integer
`seed` produces a reproducible stream; passing `None` uses OS entropy. This is preferred
over the legacy `np.random.seed` because it is thread-safe and statistically superior.

---

### 3.4 `simulate_unaffected_price_abm(seed=None)`

Arithmetic Brownian Motion: $P_{k+1} = P_k + \sigma \sqrt{dt} \, Z_k$.

ABM is the price model used in the original Almgren-Chriss (2001) paper. Under ABM, the
unaffected price can go negative, which is unrealistic for stocks but mathematically
consistent with the AC derivation. ABM simplifies the IS calculation because the drift
cancels out.

**Note**: This uses the legacy `np.random.seed(seed)` + `np.random.randn()` interface
rather than `default_rng`. This is a minor inconsistency that could cause issues in
multi-threaded environments but is harmless in the current sequential code.

---

### 3.5 `simulate_unaffected_price_gbm(mu=0.0, seed=None)`

Geometric Brownian Motion: $S_{k+1} = S_k \exp((mu - \tfrac{1}{2}\sigma^2)dt + \sigma\sqrt{dt}\,Z_k)$.

GBM keeps prices positive and is the standard Black-Scholes price model. It is included
as an alternative but is not currently used in the main simulation loop. The comment in
the code notes that ABM is used for the unaffected price in AC theory, and GBM is provided
for potential future experimentation.

---

### 3.6 `apply_market_impact(P, trades)`

This function is the cash-flow accounting engine.

**Permanent impact**:
```python
permanent_prices[k] = base_price - self.gamma * cumulative_sold
```

Every share sold before step `k` has permanently depressed the mid-price. The effect is
proportional to total shares already sold. The AC model uses *linear permanent impact*
with coefficient `gamma`.

**Temporary impact**:
```python
execution_prices[k] = permanent_prices[k] - self.eta * (trades[k] / self.dt)
```

The temporary impact penalizes trading *rate*, not trade size. Selling `n` shares over
interval `dt` moves the market by `eta * (n/dt)`. This reflects the intuition that
urgency (high rate) incurs more slippage than a slow drip of the same total quantity.

**Cash flow**:
```python
cashflows[k] = execution_prices[k] * trades[k]
```

Cash received at step `k` is the realized execution price times the number of shares sold.
Summing over all steps gives total cash received.

**Return value structure**: The function returns a dict with `permanent_prices`,
`execution_prices`, `cashflows`, and `total_cash`. The dict format (rather than a single
float) allows callers to inspect intermediate quantities for debugging without rerunning
the function.

---

### 3.7 `implementation_shortfall(X, total_cash)`

$$IS = X \cdot S_0 - \text{total\_cash}$$

This measures how much cash was lost relative to the idealized trade: if all `X` shares
had been sold instantly at the initial price `S0`. A positive IS means the execution
received less cash than the paper value; a negative IS means the price moved favorably
during execution (the Heston model under positive drift can produce this).

---

## 4. `core/Backtester.py`

The `Backtester` runs a single execution scenario: one Heston price path, one trade
schedule, full transaction logging.

### 4.1 Simulation-before-schedule ordering

```python
# Market path first
prices, variances = self.market.simulate_unaffected_price_heston(...)

# Strategy outputs conditioned on realized path
trades = self.strategy.compute_trade_list(
    use_correction=use_correction,
    variance_path=variances if use_correction else None,
)
```

This ordering is deliberate and important. The corrected strategy receives the *realized*
variance path (not the deterministic approximation). This models a trader who observes
variance in real time and adjusts their schedule accordingly. If the backtester simulated
the price path *after* computing trades, it would be using the wrong variance sequence.

### 4.2 Transaction log

Each row in `log_df` records:

| Column | Meaning |
|---|---|
| `step` | Interval index (0 to N-1, plus possible terminal row) |
| `time` | Elapsed time at start of interval |
| `unaffected_price` | Hypothetical price without market impact |
| `inventory_before` | Shares held before this trade |
| `shares_traded` | Number of shares sold in this interval |
| `inventory_after` | Shares held after this trade |
| `permanent_impact_price` | Unaffected price minus cumulative permanent impact |
| `execution_price` | The price actually received (permanent + temporary impact) |
| `cash_captured` | Shares × execution price |
| `cumulative_cash` | Running total cash received so far |

### 4.3 Forced liquidation penalty

```python
if final_inventory > 1e-8:
    final_market_price = prices[-1] - self.market.gamma * cumulative_sold
    penalty_price = final_market_price * (1.0 - forced_liquidation_discount)
    penalty_cash = final_inventory * penalty_price
```

If any shares remain at time `T`, they must be liquidated immediately at a 5% discount
(`forced_liquidation_discount=0.05`). This represents the cost of dumping a block
urgently into the market: an order-of-magnitude larger temporary impact than the
modeled slippage, modeled as a flat percentage discount.

The threshold `1e-8` rather than `0.0` avoids triggering the penalty for pure floating-point
residuals when the schedule exactly liquidates.

**In the comparator**, the forced liquidation penalty never fires, because the trade list
from `AlmgrenChrissModel` is always renormalized to sum exactly to `X`.

---

## 5. `core/MonteCarloSimulator.py`

Runs many Heston paths to estimate the distribution of IS for a given `lambda` value.

### 5.1 Monte Carlo idea

The core problem is that IS is a *random variable*: the realized price path affects how
much cash the strategy captures. A single backtest only measures IS on one path.
Monte Carlo simulation draws many paths and computes IS on each, building an empirical
distribution that supports statistical inference.

For each lambda, the code:
1. Builds the AC strategy (deterministic trade schedule for classic; pathwise for corrected).
2. Simulates `n_sims` Heston paths.
3. Computes IS for each path.
4. Returns mean, variance, and standard deviation of IS across paths.

### 5.2 Seeding strategy

```python
rng = np.random.default_rng(seed)
...
seed=int(rng.integers(0, 1_000_000_000))
```

A master RNG generates sub-seeds for each path. This makes the full Monte Carlo
reproducible from a single `seed` while ensuring each path uses an independent RNG
stream. Simply incrementing `seed + i` would work but risks correlations in some RNG
implementations.

### 5.3 `use_correction` inside the loop

When `use_correction=True`, the corrected trade schedule is recomputed on every path:

```python
if use_correction:
    trades = strategy.compute_trade_list(
        use_correction=True,
        variance_path=variance_path,
    )
```

This is necessary because the corrected schedule depends on the realized variance path,
which is different for every Monte Carlo draw. The cost is `N+1` quadrature evaluations
per path (for `compute_b_trajectory()`), which can be slow for large `N` or many simulations.
A practical optimization would be to precompute `b_values` once and pass them in, since
`b(t)` is deterministic (path-independent).

### 5.4 Parallelization of the lambda grid

```python
def run_lambda_grid(..., parallel=True, max_workers=None):
    ...
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_lambda_worker, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
```

**Why `ProcessPoolExecutor` and not `ThreadPoolExecutor`**: Python has the Global
Interpreter Lock (GIL), which prevents multiple threads from executing Python bytecode
simultaneously. For CPU-bound work like Monte Carlo simulation, `ProcessPoolExecutor`
spawns separate OS processes, each with its own Python interpreter and GIL, enabling
true parallelism.

**Why the worker function is top-level**: `ProcessPoolExecutor` uses `pickle` to serialize
tasks to worker processes. Python cannot pickle closures or lambdas, so the worker function
`_run_lambda_worker` must be defined at module level, not as a nested function or class method.

**Task structure**: Each task is a tuple `(simulator_params, lambd, n_sims, seed)`.
`simulator_params` is a plain dict (picklable), not the `MonteCarloSimulator` object itself.
The worker reconstructs the simulator inside the process. This avoids trying to pickle a
class instance with `scipy.integrate.quad` internals that may not be picklable.

**`max_workers` default**: `max(1, cpu_count - 1)` reserves one core for the OS and the
parent process, reducing interference on systems running other tasks.

**`as_completed`**: Results are collected as workers finish rather than in submission order.
This allows earlier results to be processed while slower lambda values are still running.
The final `sort_values("lambda")` restores order.

**Sequential fallback (`parallel=False`)**: Useful for debugging, since process boundaries
suppress exceptions and stack traces. In sequential mode, errors surface immediately.

---

## 6. `data/calibrator.py`

The `LobsterCalibrator` extracts model parameters from real market microstructure data.

### 6.1 LOBSTER data format

LOBSTER (Limit Order Book System — The Efficient Reconstructor) produces two paired CSVs:
- **Message file**: Every order book event (submission, cancellation, execution) timestamped
  to nanoseconds.
- **Orderbook file**: A snapshot of the limit order book after each event, with price and
  size at each level.

LOBSTER prices are stored as integers scaled by 10000 (to avoid floating-point in the
original data):
```python
orderbook[price_cols] = orderbook[price_cols] / self.PRICE_SCALE
```

### 6.2 Dataset constructors

Three class methods (`from_zip`, `from_directory`, `from_dataset`) provide a unified
interface regardless of whether data is in a zip archive or an extracted directory.

**Automatic level detection**:
```python
level_match = re.search(r"_(\d+)\.csv$", orderbook_file)
if level_match:
    levels = int(level_match.group(1))
```

LOBSTER filenames encode the number of price levels in the orderbook (e.g., `_10.csv`
means 10 levels). The regex extracts this. As a fallback, the number of columns is counted:
`levels = len(first_line.split(",")) // 4` (4 columns per level: ask price, ask size,
bid price, bid size).

### 6.3 `load_data()`

**Mid price**:
```python
df["Mid_Price"] = (df["Ask_Price_1"] + df["Bid_Price_1"]) / 2.0
```
The best bid-ask midpoint is the standard unaffected price proxy.

**Micro price**:
```python
df["Micro_Price"] = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
```
The microprice is a depth-weighted mid: if the ask side has less depth, the price is
"pulled" toward the ask, reflecting information about near-term price pressure. This is
more predictive of short-term price moves than the simple mid, but we primarily use mid.

**Data cleaning**:
```python
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Time", "Mid_Price", "Ask_Price_1", "Bid_Price_1"])
df = df[df["Mid_Price"] > 0].copy()
```
Infinities can appear when dividing by zero (e.g., zero bid size). Rows with missing
key columns or non-positive mid prices are dropped before any analysis.

---

### 6.4 `estimate_volatility(df, freq="5min")`

```python
intervals_per_day = self._intervals_per_day(freq)   # = 78 for 5min
daily_vol = log_returns.std(ddof=1) * np.sqrt(intervals_per_day)
```

This applies the square-root-of-time rule for volatility scaling. Variance scales linearly
with time for i.i.d. returns, so variance per interval multiplied by `intervals_per_day`
gives daily variance, and taking the square root gives daily volatility.

**Log returns** (`_log_returns`): Mid-prices are resampled to 5-minute bars (taking the
last mid-price in each bar). Log returns `log(P_t / P_{t-1})` are approximately normally
distributed for small changes and are preferred over simple returns because they are
additive and bounded below by -1.

`ddof=1` applies Bessel's correction for sample standard deviation (uses `n-1` in the
denominator rather than `n`), appropriate for estimating the population parameter from a
finite sample.

---

### 6.5 `_estimate_volatility_series(df, window="1min")`

Produces a time series of short-window rolling volatilities for Heston parameter estimation.

```python
lo, hi = returns.quantile([0.001, 0.999])
returns = returns.clip(lower=lo, upper=hi)
vol_series = returns.rolling(window=10, min_periods=3).std()
```

**Clipping**: The 0.1th and 99.9th percentile clip removes extreme microstructure spikes
(quote stuffing, data errors, crossed markets). These outliers would severely distort
the rolling variance estimates.

**Rolling std with `window=10, min_periods=3`**: Uses a 10-period window for each
volatility estimate. `min_periods=3` allows estimates near the start of the series
where fewer than 10 observations are available, avoiding excessive `NaN` masking.

**Why 1-minute**: 1-minute bars are short enough to track intraday variance variation
(needed for Heston dynamics) but long enough to suppress microstructure noise.

---

### 6.6 Variance rescaling to daily units

```python
intervals_per_day = self._intervals_per_day(window)   # = 390 for 1min
var_series = var_series * intervals_per_day
```

This is the critical unit conversion. The rolling standard deviation is computed from
1-minute log returns, so it has units of variance per 1-minute interval. Multiplying by
390 converts to daily variance, consistent with the `T = 1 = one day` convention used
throughout the model.

Without this rescaling, the Heston parameters `v0` and `omega` would be ~390× too small,
making the correction term negligibly small regardless of the market regime.

---

### 6.7 `estimate_impact_parameters(df)`

Filters to LOBSTER event types 4 and 5 (market buy and market sell orders respectively):
```python
market_orders = df[df["Event"].isin([4, 5])].copy()
```

These are the events that actually consume liquidity from the book.

**Temporary impact estimation (`_estimate_temporary_impact`)**:

The idea is to simulate what would happen if the trader placed a market order of size `X`
against the current orderbook. The "walk the book" procedure:

```python
for level in range(1, self.levels + 1):
    fill = min(remaining, float(size))
    total_cash += fill * float(price)
    total_filled += fill
    remaining -= fill
```

For a given trade size `X`, fills are consumed level by level at increasing prices.
If `remaining > 0` after exhausting all levels, the order cannot be filled and `None`
is returned. The average fill price minus the mid price gives the market impact.

This is repeated across 20 trade sizes (from 10th to 80th percentile of total ask depth)
and a sample of 1000 orderbook snapshots. The resulting (size, impact) pairs are fitted
to a linear model through the origin:

$$\text{impact} = \eta \cdot X \qquad \Rightarrow \qquad \eta = \frac{\sum X_i \cdot \text{impact}_i}{\sum X_i^2}$$

This is ordinary least squares without intercept, computed analytically.

**Permanent impact estimation (`_estimate_permanent_impact`)**:

Signed order flow is defined as:
- `+size` for direction=1 (buy) market orders
- `-size` for direction=-1 (sell) market orders

5-minute aggregated signed flow is used to reduce noise. The permanent impact is then:

$$\Delta P_t = \gamma \cdot \text{signed\_flow}_{t-1} + \epsilon$$

where the lagged flow predicts the future mid-price change. `np.polyfit(x, y, 1)` performs
ordinary least squares and `gamma = abs(slope)`. The absolute value is taken because the sign
is absorbed into the direction convention — the model always assumes gamma reduces the price
for a seller.

---

### 6.8 Heston parameter estimation pipeline

```python
kappa, omega, xi, rho = self._estimate_volatility_parameters(df, var_series)
```

**Mean reversion speed `theta` and `omega` via OLS**:

The Heston variance SDE discretized is:

$$\Delta v_t = \theta(\omega - v_t) \Delta t + \xi \sqrt{v_t \Delta t} \, Z$$

Taking expectations: $E[\Delta v_t] \approx \theta \omega \Delta t - \theta v_t \Delta t$.
This is a linear regression:

$$\Delta v_t = a + b \cdot v_t \qquad \text{with} \quad a = \theta \omega \, dt, \quad b = -\theta \, dt$$

```python
X = np.column_stack([np.ones_like(v_t), v_t])
beta, *_ = np.linalg.lstsq(X, dv, rcond=None)
a, b = beta
kappa = -b / dt
omega_from_regression = a / (kappa * dt)
```

`np.linalg.lstsq` is used rather than `np.polyfit` for numerical robustness with the
design matrix structure.

**Vol-of-vol `xi` from residuals**:

After computing the deterministic drift, residuals are:

$$\text{residual}_t = \Delta v_t - \theta(\omega - v_t) \Delta t$$

Theoretically, $\text{residual}_t \approx \xi \sqrt{v_t \Delta t} \, Z$, so:

$$\xi \approx \text{std}\!\left(\frac{\text{residual}_t}{\sqrt{v_t \Delta t}}\right)$$

```python
denom = np.sqrt(np.maximum(v_t * dt, 1e-12))
xi_samples = residuals / denom
xi = float(np.nanstd(xi_samples, ddof=1))
```

The `maximum(v_t * dt, 1e-12)` prevents division by zero for very small variance values.

**Correlation `rho`**:

```python
rho = returns.loc[common_index].corr(var_changes.loc[common_index])
```

The Pearson correlation between 1-minute log returns and 1-minute variance changes
estimates the correlation parameter. Negative values indicate the leverage effect.

**Safety bounds**:

```python
params["v0"] = float(np.clip(params["v0"], 1e-12, 1.0))
params["theta"] = float(np.clip(params["theta"], 1e-6, 50.0))
params["omega"] = float(np.clip(params["omega"], 1e-12, 1.0))
params["xi"] = float(np.clip(params["xi"], 1e-6, 5.0))
params["rho"] = float(np.clip(params["rho"], -0.95, 0.95))
```

OLS on noisy financial data can produce nonsensical values (negative variance, arbitrarily
large mean-reversion speeds). Clipping enforces plausible ranges derived from the financial
economics of the Heston model.

**Feller condition enforcement**:

$$2\theta\omega > \xi^2$$

The Feller condition guarantees that the Heston variance process stays strictly positive
almost surely. Without it, the discretized variance can hit zero and stay there ("absorbed"),
producing degenerate flat paths. Rather than rejecting the parameters, `xi` is reduced
to just below the Feller boundary:

```python
adjusted_xi = 0.95 * np.sqrt(max(feller_lhs, 1e-12))
```

The 0.95 factor provides a safety margin.

---

## 7. `evaluation/comparator.py`

Orchestrates a paired statistical comparison between the classic AC strategy and the
Heston-corrected strategy.

### 7.1 Paired comparison design

Both strategies are evaluated on the **same** simulated Heston price path in each trial.
This eliminates path noise from the comparison: any difference in IS between the two
strategies on a given path is purely attributable to the difference in the trade schedule,
not to random variation in the path.

This is the correct experimental design for a paired test. It was a key fix from an
earlier version that used separate independently-drawn paths for each strategy, making the
paired t-test statistically invalid.

### 7.2 `__init__` — syncing Heston parameters to the AC model

```python
if hasattr(model_hest, 'xi'):
    self.model_ac.xi = model_hest.xi
if hasattr(model_hest, 'rho'):
    self.model_ac.rho = model_hest.rho
```

The `HestonParameters` object passed as `model_hest` carries the simulation parameters.
This sync ensures the AC model's correction term uses the same `xi` and `rho` as the
price simulator, maintaining parameter consistency.

### 7.3 `_simulate_heston_path(path_seed, path_index)`

Generates one shared path and applies sanity checks:

```python
if np.nanmax(price_path) > self.market_env.S0 * 2.0 or \
   np.nanmin(price_path) < self.market_env.S0 * 0.2:
    # discard
```

A path where the price doubles or falls by 80% over one day is almost certainly a
numerical artifact from the Euler discretization with extreme variance paths. These
paths are discarded rather than filtered post-hoc because they would dominate the IS
distribution and obscure the comparison signal.

The regime volatility metric `regime_vol = mean(sqrt(max(variance_path, 0)))` is the
mean realized volatility over the path, used for regime bucketing in the statistical tests.

### 7.4 `run_comparison()` — full paired loop

```python
for i in range(self.num_sims):
    path_seed = int(rng.integers(0, 1_000_000_000))
    price_path, variance_path, regime_vol = self._simulate_heston_path(path_seed, i)

    if price_path is None:
        continue

    # Baseline: static AC trades on this path
    is_ac[i] = self._evaluate_trades_on_price_path(trades_classic, price_path, ...)

    # Challenger: corrected trades computed from this path's variance
    corrected_trades = self.model_ac.compute_trade_list(
        use_correction=True,
        variance_path=variance_path,
    )
    is_heston[i] = self._evaluate_trades_on_price_path(corrected_trades, price_path, ...)
```

Note that `trades_classic` is precomputed once (it is deterministic), while `corrected_trades`
is recomputed each iteration because it depends on the realized `variance_path`.

---

## 8. `evaluation/statistics.py`

Implements a multi-test statistical battery comparing the IS distributions of both strategies.

### 8.1 Motivation for multiple tests

No single test is sufficient:
- The **paired t-test** is optimal for detecting mean differences but assumes normality and is
  sensitive to outliers.
- The **Wilcoxon signed-rank test** is the non-parametric alternative for the median; robust
  to fat tails.
- The **Levene test** checks whether the Heston strategy has *lower variance* (i.e., more
  predictable execution cost), which is a risk-management question distinct from mean IS.
- The **KS test** checks whether the entire distributions differ, not just mean or variance.
- **CVaR** evaluates tail risk: does the Heston strategy avoid catastrophic execution
  outcomes in the worst 5% or 1% of scenarios?
- **Regime analysis** asks whether Heston outperforms specifically when conditions are
  favorable (e.g., high-volatility regimes where dynamic adjustment matters most).

### 8.2 Paired t-test

$$H_0: \mu(\text{IS}_{AC} - \text{IS}_{Hest}) = 0 \qquad H_1: \mu > 0 \text{ (Heston better)}$$

```python
t_stat, p_two = ttest_rel(is_ac, is_hest)
p_ttest = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
```

`ttest_rel` gives the two-sided p-value. To get the one-sided p-value for the hypothesis
that AC IS is *greater* than Heston IS (Heston better):
- If `t_stat > 0`, the observed difference is in the right direction; one-sided p = two-sided p / 2.
- If `t_stat <= 0`, the evidence is in the wrong direction; one-sided p = 1 - two-sided p / 2.

### 8.3 Wilcoxon signed-rank test

The Wilcoxon test ranks the absolute differences and checks whether the signed ranks are
centered at zero. It is equivalent to a median test for paired samples under symmetry.
`alternative="greater"` specifies that we test for positive median difference
(AC IS > Heston IS → Heston better).

```python
non_zero_diff = diff[diff != 0]
if len(non_zero_diff) == 0 or np.allclose(diff, 0):
    # Skip: test undefined when all differences are zero
```

If both strategies produce identical IS on every path (which happens when the correction
collapses to AC), the Wilcoxon test cannot be computed — all differences are zero, so no
signed ranks exist.

### 8.4 Levene's test for variance equality

```python
lev_stat, p_lev_two = levene(is_ac, is_hest)
p_levene = p_lev_two / 2 if np.var(is_ac) > np.var(is_hest) else 1.0 - p_lev_two / 2
```

Levene's test is robust to non-normality (unlike Bartlett's test). One-sided conversion
uses the same logic as the t-test: if `var(AC) > var(Hest)` (Heston less variable), the
one-sided p-value is half the two-sided value.

### 8.5 Kolmogorov-Smirnov test

The KS statistic is the maximum absolute difference between the empirical CDFs of the
two IS distributions. A significant result means the distributions are different in *shape*
somewhere, not necessarily in mean or variance. The two-sided alternative checks for any
difference.

### 8.6 CVaR and bootstrap inference

**CVaR definition**:
```python
def _cvar(losses, alpha=0.95):
    var = np.quantile(losses, alpha)
    return float(np.mean(losses[losses >= var]))
```

CVaR (also called Expected Shortfall) at level `alpha` is the expected IS conditional on
being in the worst `1-alpha` fraction. It is a coherent risk measure, preferred over VaR
for risk management because it is subadditive and captures tail shape.

**Bootstrap confidence interval and p-value**:

```python
for i in range(n_bootstrap):
    idx = rng.integers(0, n, size=n)
    boot_diffs[i] = _cvar(hest[idx], alpha) - _cvar(ac[idx], alpha)
p_value = float(np.mean(boot_diffs <= obs_diff))
```

10,000 bootstrap resamples (with replacement) of the *paired* sample build the empirical
distribution of the CVaR difference. The p-value counts how often the bootstrap differences
are at least as extreme as the observed difference, testing H0: `CVaR_Hest >= CVaR_AC`.
The 95% CI is the 2.5th–97.5th percentile of the bootstrap distribution.

**Why bootstrap instead of an asymptotic test**: CVaR does not have a simple normal
asymptotic distribution (especially with heavy-tailed IS data), making analytical p-values
unreliable. The bootstrap is distribution-free and works well here.

### 8.7 Regime analysis

```python
sorted_idx = np.argsort(regime_values, kind="mergesort")
regime_splits = np.array_split(sorted_idx, n_regimes)
```

Paths are sorted by the regime metric (mean path volatility) and split into `n_regimes`
equal-size buckets. Within each bucket, a one-sided Wilcoxon signed-rank test tests
whether Heston outperforms AC. This tests the hypothesis that the Heston correction is
most beneficial when variance is high.

**`mergesort`** is used for the stable sort to preserve original ordering among ties,
ensuring deterministic bucket assignment.

### 8.8 Leverage sensitivity (`rho_sweep`)

When a `rho_sweep` dict is provided (mapping `rho → {'is_ac': ..., 'is_hest': ...}`):

```python
sp_r, sp_p = stats.spearmanr(rhos, diffs)
```

Spearman's rank correlation (not Pearson) is used because the relationship between `rho`
and IS difference may be monotone but not linear. A significant negative Spearman
correlation would mean: the more negative `rho` (stronger leverage effect), the larger
Heston's advantage.

### 8.9 Figures

Three matplotlib figures are produced:
1. **Distribution overview**: IS histograms for both strategies, paired-difference histogram,
   and CVaR bar chart at 95%/99%.
2. **Regime analysis**: Bar chart of median IS difference per volatility regime, colored
   green (Heston better) or red (Heston worse), with p-values annotated.
3. **Leverage sensitivity**: Line plots of mean IS and CVaR 95% as a function of `rho`,
   with Spearman correlation in the subtitle.

### 8.10 `print_results()`

A structured pretty-printer for the nested results dictionary, used by the main menu to
show results after a model comparison run. It prints each test section with a header separator,
handles missing keys gracefully, and formats numbers consistently.

---

## 9. `evaluation/correction_term_diagnostics.py`

A standalone diagnostic script that quantifies whether the Heston correction term is
producing economically meaningful trade adjustments.

### 9.1 `run_scenario_sensitivity(base, scenarios, n_sims=300)`

Tests three scenarios:
- **Equilibrium**: `v0 = omega` — variance starts at its long-run mean. Correction
  expected to be small since paths jitter symmetrically around the mean.
- **Shock-1**: `v0 = 4 * omega` — large initial variance spike. The correction should
  front-load selling to avoid executing during high-variance early periods.
- **Shock-2**: `v0 = 4 * omega` with `omega = 0.03` — same spike but lower long-run mean,
  testing robustness to out-of-equilibrium starts.

For each scenario, prints maximum and mean absolute `delta_trade` (change in trade size
vs AC baseline) and the mean IS improvement.

### 9.2 `run_inventory_scaling_probe(base, hparams, seed=7)`

Tests three variants of the correction scaling:

- **A-current**: The live implementation (correction chunks scaled by `current_x`).
- **B-xfrac**: Correction chunks scaled by `current_x / X` (fractional remaining inventory).
- **C-xraw**: Correction chunks scaled by `current_x` applied *outside* the loop (reference).

Comparing A vs B vs C on the same path and variance stream isolates the effect of the
inventory multiplier.

### 9.3 `sys.path` injection

```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

Since the script lives in `evaluation/` rather than the project root, it cannot import
`core.*` by default. This adds the project root to the Python module search path at
runtime, allowing direct execution with `python evaluation/correction_term_diagnostics.py`.

---

## 10. `main.py`

The interactive entry point. Provides a menu-driven interface for all major workflows.

### 10.1 `HestonParameters` class

```python
class HestonParameters:
    def __init__(self, v0, mu, theta, omega, xi, rho): ...
```

A simple data class carrying Heston parameters as attributes. The `ModelComparator`
accesses these via attribute access (`h.v0`, `h.xi`, etc.). This is intentionally kept
simple rather than using a `dataclass` or `NamedTuple` to avoid import overhead and for
compatibility with `comparator.py`'s `hasattr` checks.

### 10.2 `build_objects(params, lambd)`

Central factory that builds the four primary objects from a parameter dict and a lambda:

```python
return strategy, env, sim, back
```

- `strategy`: `AlmgrenChrissModel` with all Heston parameters forwarded.
- `env`: `MarketEnvironment` with `heston_params` stored for the backtester.
- `sim`: `MonteCarloSimulator` for lambda optimization.
- `back`: `Backtester` wrapping strategy and env.

Having all construction in one place ensures parameter consistency — there is no risk of
`strategy` and `env` using different `eta` values, for example.

### 10.3 `list_datasets()` and `is_lobster_directory()`

```python
def is_lobster_directory(path):
    has_message = any("_message_" in child and child.lower().endswith(".csv") ...)
    has_orderbook = any("_orderbook_" in child and child.lower().endswith(".csv") ...)
    return has_message and has_orderbook
```

Only directories containing both a LOBSTER message file and an orderbook file are
returned. This prevents `__pycache__/` and other non-data directories from appearing
in the dataset selection menu.

The function prefers extracted directories over zip files with the same base name (using
the `seen_bases` set), avoiding duplicates when a zip has already been extracted.

### 10.4 `unzip_datasets_once()`

Automatically extracts zip archives on first run and writes a marker file `.unzipped_once`
to prevent re-extraction:

```python
if os.path.exists(marker_path):
    return
```

This avoids repeatedly unzipping large archives on every run. The marker is checked before
any zip scanning.

### 10.5 Lambda optimization — two-stage grid search

```python
# Stage 1: coarse
coarse_lambda_values = np.linspace(min_lambda, max_lambda, num_values)
coarse_results = sim.run_lambda_grid(coarse_lambda_values, ...)
best_lambda = coarse_results.loc[coarse_results["objective"].idxmin()]["lambda"]

# Stage 2: refine around best
refine_width = (max_lambda - min_lambda) / num_values
refined_min = max(min_lambda, best_lambda - refine_width)
refined_max = min(max_lambda, best_lambda + refine_width)
refined_lambda_values = np.linspace(refined_min, refined_max, num_values)
```

**Two-stage search rationale**: Running 1000 simulations for 20 lambda values is
computationally feasible. But using all those simulations on a uniform grid over a
wide range wastes resolution near the optimum. The two-stage approach spends stage 1
finding the rough location of the optimum and stage 2 spending the same computation
budget on a narrow region around it.

**Objective function**:
```python
results["objective"] = results["mean_is"] + risk_penalty * results["std_is"]
```

The objective trades off mean IS (want low) against variance of IS (want low). The
`risk_penalty` (default 1.0) weights the variability penalty. A pure minimizer of
mean IS would choose very high `lambda` (aggressive), but high `lambda` also produces
a highly variable schedule. The objective balances these.

### 10.6 `run_dataset_sweep()` — cross-dataset validation

The sweep runs the full AC vs Heston comparison across every available dataset using
parallel workers. Each dataset produces its own calibrated parameters, so the comparison
is across heterogeneous market conditions (different stocks, different intraday liquidity).

**Across-dataset paired t-test**:
```python
t_stat, p_two = ttest_rel(mean_ac_valid, mean_hest_valid)
p_ttest = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
```

This tests whether the Heston model systematically outperforms AC across datasets, treating
each dataset's mean IS as one observation in a paired test. This controls for within-dataset
correlations, since both strategies are run on the same paths per dataset.

**Sharpe-like ratio**:
```python
def calculate_sharpe_like(mean_is: float, std_is: float) -> float:
    return float(-mean_is / std_is)
```

Defined as `-mean_is / std_is`. A more negative mean IS (better) and smaller std IS (more
reliable) both increase this ratio. It is analogous to a Sharpe ratio but for implementation
shortfall: higher values indicate better risk-adjusted execution quality.

The sweep also runs a Sharpe-ratio comparison across datasets to test whether the Heston
strategy is not just lower in expected cost but also more *reliable*.

### 10.7 Worker function pickling requirement

```python
def run_one_dataset_sweep_task(args):
    """
    Worker function for one dataset.
    This must be top-level so ProcessPoolExecutor can pickle it on Windows.
    """
```

As with `_run_lambda_worker`, this must be a top-level function. On Windows, `multiprocessing`
uses `spawn` (not `fork`) to create worker processes, which requires pickling everything
sent to the worker. Methods and closures are not picklable in Python.

On Linux/macOS, `fork` is used by default and avoids pickling the worker function, but
`spawn` can still be triggered (e.g., within `if __name__ == '__main__':` guards).
Keeping the function at module level ensures cross-platform safety.

---

## 11. Cross-Cutting Engineering Decisions

### 11.1 RNG strategy

The project uses two RNG interfaces:

- **`np.random.default_rng(seed)`**: Used throughout the Heston simulation and Monte Carlo
  loops. This is the modern NumPy Generator API (PCG64 algorithm), which is thread-safe,
  statistically superior to the Mersenne Twister, and produces reproducible streams with
  integer seeds.

- **`np.random.seed()` + `np.random.randn()`**: Still present in `simulate_unaffected_price_abm`
  and `simulate_unaffected_price_gbm`. This is the legacy global state RNG, which is not
  thread-safe. These functions are not used in any parallel or multi-call loop, so this is
  harmless but should be updated for consistency.

### 11.2 Exact liquidation guarantee

Every path in the code that produces a trade schedule ensures total traded shares equals
exactly `X`:

- **Classic AC**: `sum(x[:-1] - x[1:]) = x[0] - x[N] = X - 0 = X` (exact analytically).
- **Corrected AC**: Renormalized by `corrected_trades *= X / total_sold` after the loop.

This is a hard constraint of the problem: all shares must be sold by `T`.

### 11.3 Non-negativity of trades

The AC closed-form trajectory is monotone decreasing, so `n_k = x_{k-1} - x_k >= 0` always.
The corrected schedule adds a signed perturbation that could in principle make some trade
negative (i.e., buy back shares). The `max(proposed_trade, 0.0)` clamp enforces the
no-short-selling constraint.

### 11.4 Parameter propagation consistency

All Heston parameters flow from the user (via `params["heston"]`) through `build_objects`
into all four objects (`AlmgrenChrissModel`, `MarketEnvironment`, `MonteCarloSimulator`,
`Backtester`). The `MarketEnvironment` stores `heston_params` as an attribute so the
`Backtester` can simulate Heston paths with the same parameters as the strategy's correction term.

### 11.5 Discretization choice: left-endpoint variance

In `_compute_trade_correction_chunks`, the variance at step `k` uses `variance_path[k]`,
which is the *left endpoint* of the interval `[t_k, t_{k+1}]`. A midpoint rule
`0.5*(variance_path[k] + variance_path[k+1])` would be more accurate for integration,
but tests showed the difference in IS impact is negligible (< 0.002 shares across 200 paths).
The left-endpoint convention is simpler and requires no lookahead.

### 11.6 Feller condition as a soft guard

Violating the Feller condition (`2*theta*omega <= xi^2`) means variance can reach zero
and potentially stay there in the discrete-time simulation. Rather than rejecting parameter
sets (which could cause silent failures in the sweep), the calibrator *reduces* `xi` to
just below the Feller boundary. This is logged as a warning so the user knows the
calibration was adjusted.

### 11.7 Why `scipy.integrate.quad` and not a closed-form integral

The integral in `compute_b_value` involves $e^{-\theta(s-t)} \cdot \sinh^2(\kappa(T-s)) / \sinh^2(\kappa(T-t))$.
A closed-form antiderivative exists but requires case analysis for near-zero `theta` and `kappa`,
and the resulting formula is complex enough that coding errors are likely. `quad` with
adaptive quadrature is accurate to ~1e-10 in these smooth integrands and much safer.
The performance overhead is acceptable for the current N=78 grid.
