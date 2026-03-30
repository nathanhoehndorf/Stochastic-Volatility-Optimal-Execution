# MATH310_OptimalTradingRate

Figure out what we actually need to do based off of Gemini outlines, since neither is likely to be super accurate. Here is my attempt:
1. Lock down exact algebraic formula for the derivative of $J^{(0)}$.
2. Discretize Heston using Euler-Maruyama
3. Write `generate_heston_path` function
4. Create mechanism that adjusts synthetic prices based on agent's actions, incorporating permanent and temporary market impact
5. Code Static AC Agent (calculate trajectory upfront)
6. Code Dynamic Agent (calculate at each time step)
7. Build execution loop
  a. Go through steps $k=1,\dots, N$
  b. Create transaction log (pandas df) to record time, remaining inventory, execution price, and cash captured at every step
  c. Calculate IS
8. Run trade lists through to test simulations
9. Plot mean IS to variance of IS.

## Gemini Outline 1:
### 1. Build the Heston Market Simulator
Before testing, we need a simulation world for the executation strategy to live in. We are going to discretize the Heston SDE using a Euler-Maruyama scheme. Using this, we can generate synthetic price paths $(S_t)$ and variance paths $(v_t)$.
- Convert $dv_t=\theta(\omega-v_t)dt+\xi\sqrt{v_t}dW_t$ into $v_{t+\Delta t}=v_t+\theta(\omega-v_t)\Delta t+\xi\sqrt{v_t\Delta t}\cdot Z_1$
- Use a Cholesky decomposition to ensure price noise $Z_2$ and voltatility noise $Z_1$ are correlated by some $\rho$.
- Use numpy to generate paths
- We'll need a function `generate_heston_path(S0, v0, params, T, steps)` that returns two arrays: prices and variances
### 2. Encode The Base and Correction Functions
The solution relies on the derivative $\frac{\partial^2 J^{(0)}}{\partial x \partial v}$. We need to hardcode the analytical $J^{(0)}$ and its derivatives so the computer and compute the optimal control every time step.
- Define the Almgren-Chriss kernel by writing a function for the hyperbolic trig terms
- Control Loop: at every time step t:
  - Observe current inventory $x_t$ and current volatility $v_t$
  - Calculate $n_{static}$
  - Calculate the stochastic adjustment using the $J^{(1)}$ logic
  - The final trade size $n^*=n_{static}+\text{Correction}$
  ### 3. Create the Backtester
  This is where trading simulation happens.
  - Transaction Log: Create a stucture (pandas df) to record Time (t), Remaining Inventory ($x_k$), Execution Price ($\tilde{S_k}=S_k-\eta n_k-\dots$), Cash Captured
  - Penalty: At t=T, if $x_T>0$, apply a "liquidate at all costs" to final shortfall.
  ### 4. Statistics
  We now have a simulator. Run 10,000 to do a Monte Carlo, and track all of the statistics we were talking about like expected implementation shortfall, value at risk (95th percentile of worst outcomes), and utility comparisons.

  We should also do visualizations of all of our stuff

  ## Gemini Outline 2:
### 1. Implement Core Math Model
1. Create a class/function that takes the core parameters:
- X: total shares
- T: Total time
- N: number of trading intervals
- $\sigma$: Asset volatility
- $\lambda$: Risk aversion
- $\eta$: Temporary market impact coefficient
- $\gamma$: Permanent market impact coefficient
2. Calculate Trajectory
Compute $\kappa=\sqrt{\frac{\lambda\sigma^2}{\eta}}$ and $x_t=X\frac{\sinh{(\kappa(T-t))}}{\sinh{(\kappa T)}}$
3. Calculate number of shares to trade at each step, $n_k=x_{k-1}-x_k$.
### 2. Build Market Environment
1. Simulate Unaffected price
Write a NumPy or SymPy generator that produces an unaffected price path using Arithmetic Brownian Motion (used in Almgren-Chriss) or Geometric Brownian Motion: $P_k=P_{k-1}+\sigma\sqrt{\tau}Z_k$, with $Z_k$ standard normal.
2. Incorporate Market Impact
Create a function that calculates actual execution price via Permanent impact $P_k=P_{k-1}-\gamma n_k$ and Temporary Impact $\tilde{P_k}=P_k-\eta\frac{n_k}{\tau}$
3. Calculate IS
Calculate difference between initial value of portfolio and the cash received.
### 3. Monte Carlo Simulations
1. Run trade list through synthetic market thousands of times
2. For a specific $\lambda$, plot the distribution of IS, which should be normal
3. Run Monte Carlo across different $\lambda$ values. Plot mean implementation shortfall against variance of shortfall.
