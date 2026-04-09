import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
def mc_worker(args): #used for the parallel batching of the montecarlo simulations
    S0, X, T, N, sigma, eta, gamma, lambd, n_sims, seed = args

    rng = np.random.default_rng(seed)
    dt = T / N
    sqrt_dt = np.sqrt(dt)

    times = np.linspace(0, T, N + 1)
    kappa = np.sqrt((lambd * sigma**2) / eta)

    if np.isclose(kappa, 0):
        inventory = X * (1 - times / T)
    else:
        inventory = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

    trades = inventory[:-1] - inventory[1:]
    cumulative_sold_before = np.concatenate([[0], np.cumsum(trades[:-1])])

    Z = rng.standard_normal((n_sims, N)) # is this the actual distribution that we want to be using?
    dP = sigma * sqrt_dt * Z

    P = np.zeros((n_sims, N + 1))
    P[:, 0] = S0
    P[:, 1:] = S0 + np.cumsum(dP, axis=1)

    P_start = P[:, :-1]
    perm_prices = P_start - gamma * cumulative_sold_before
    exec_prices = perm_prices - eta * (trades / dt)

    cashflows = exec_prices * trades
    total_cash = np.sum(cashflows, axis=1)

    return X * S0 - total_cash

# this is the parallel batching of the montecarlo simulations to hopefully speed up the process a decent bit
def monte_carlo_is_parallel(S0, X, T, N, sigma, eta, gamma, lambd,
                            n_sims=1000, n_workers=4, seed=42): # likely don't want seed hard coded
    sims_per_worker = [n_sims // n_workers] * n_workers
    for i in range(n_sims % n_workers):
        sims_per_worker[i] += 1

    args_list = []
    for i, chunk in enumerate(sims_per_worker):
        args_list.append((S0, X, T, N, sigma, eta, gamma, lambd, chunk, seed + i))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(mc_worker, args_list))

    return np.concatenate(results)

def generate_heston_path(S0, v0, params, T, steps, seed=None):
    """
    Generate one Heston price path and one variance path using Euler-Maruyama.

    Parameters
    ----------
    S0 : float
        Initial asset price
    v0 : float
        Initial variance
    params : dict
        Must contain:
            mu    : drift of price
            theta : mean reversion speed of variance
            omega : long-run variance level
            xi    : vol of vol
            rho   : correlation between price and variance shocks
    T : float
        Total time horizon
    steps : int
        Number of time steps
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    S : np.ndarray
        Simulated price path of length steps + 1
    v : np.ndarray
        Simulated variance path of length steps + 1
    """
    if seed is not None:
        np.random.seed(seed)

    mu = params["mu"]
    theta = params["theta"]
    omega = params["omega"]
    xi = params["xi"]
    rho = params["rho"]

    dt = T / steps
    sqrt_dt = np.sqrt(dt)

    # Arrays to store the path
    S = np.zeros(steps + 1)
    v = np.zeros(steps + 1)

    S[0] = S0  #setting the first element of the arrays to the initial variance
    v[0] = v0

    # Correlation matrix for [Z1, Z2]
    corr_matrix = np.array([
        [1.0, rho],
        [rho, 1.0]
    ])

    # Cholesky factor
    L = np.linalg.cholesky(corr_matrix) #numpy library really doing some heavy lifting here
    # returns either a lowerr or uppoer triangular matrix

    for t in range(steps):
        # Generate two independent standard normals
        z = np.random.randn(2) # do we want normal distributions here or a different distribution?

        # Correlate them
        z_corr = L @ z
        Z1 = z_corr[0]   # variance shock
        Z2 = z_corr[1]   # price shock

        # Keep variance nonnegative inside sqrt
        v_t = max(v[t], 0.0)

        # Variance update
        v[t + 1] = v[t] + theta * (omega - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Z1

        # Optional truncation so variance never goes negative
        v[t + 1] = max(v[t + 1], 0.0)

        # Price update
        S[t + 1] = S[t] + mu * S[t] * dt + np.sqrt(v_t) * S[t] * sqrt_dt * Z2

    return S, v

class AlmgrenChrissModel:
    def __init__(self, X, T, N, sigma, lambd, eta, gamma):
        """
        Core Almgren-Chriss execution model.

        Parameters
        ----------
        X : float
            Total number of shares to execute
        T : float
            Total trading horizon
        N : int
            Number of trading intervals
        sigma : float
            Asset volatility
        lambd : float
            Risk aversion parameter
        eta : float
            Temporary market impact coefficient
        gamma : float
            Permanent market impact coefficient
        """
        self.X = X
        self.T = T
        self.N = N
        self.sigma = sigma
        self.lambd = lambd
        self.eta = eta
        self.gamma = gamma

        self.dt = T / N
        self.times = np.linspace(0, T, N + 1)

    def compute_kappa(self):
        """
        Compute kappa = sqrt(lambda * sigma^2 / eta)
        """
        return np.sqrt((self.lambd * self.sigma**2) / self.eta)

    def compute_inventory_trajectory(self):
        """
        Compute optimal remaining shares x_t at each grid point.
        Returns array of length N+1.
        """
        kappa = self.compute_kappa()

        # Handle kappa near zero to avoid division problems
        if np.isclose(kappa, 0):
            # Linear liquidation limit
            return self.X * (1 - self.times / self.T)

        numerator = np.sinh(kappa * (self.T - self.times))
        denominator = np.sinh(kappa * self.T)

        x = self.X * numerator / denominator
        return x

    def compute_trade_list(self):
        """
        Compute shares traded each step:
        n_k = x_{k-1} - x_k

        Returns array of length N.
        """
        x = self.compute_inventory_trajectory()
        n = x[:-1] - x[1:]
        return n

    def summary(self):
        """
        Return all major outputs in a dictionary.
        """
        x = self.compute_inventory_trajectory()
        n = self.compute_trade_list()
        kappa = self.compute_kappa()

        return {
            "kappa": kappa,
            "times": self.times,
            "inventory": x,
            "trades": n,
            "dt": self.dt
        }

class MarketEnvironment:
    def __init__(self, S0, sigma, T, N, gamma, eta):
        """
        Market environment for Almgren-Chriss style execution.

        Parameters
        ----------
        S0 : float
            Initial unaffected price
        sigma : float
            Volatility parameter
        T : float
            Total trading horizon
        N : int
            Number of intervals
        gamma : float
            Permanent market impact coefficient
        eta : float
            Temporary market impact coefficient
        """
        self.S0 = S0
        self.sigma = sigma
        self.T = T
        self.N = N
        self.gamma = gamma
        self.eta = eta
        self.dt = T / N
        self.sqrt_dt = np.sqrt(self.dt)

    def simulate_unaffected_price_abm(self, seed=None):
        """
        Simulate unaffected price path using Arithmetic Brownian Motion:
            P_k = P_{k-1} + sigma * sqrt(dt) * Z_k

        Returns
        -------
        P : np.ndarray
            Unaffected price path of length N+1
        """
        if seed is not None:
            np.random.seed(seed)

        P = np.zeros(self.N + 1)
        P[0] = self.S0

        for k in range(1, self.N + 1):
            Zk = np.random.randn()
            P[k] = P[k - 1] + self.sigma * self.sqrt_dt * Zk

        return P

    def simulate_unaffected_price_gbm(self, mu=0.0, seed=None): # I highly doubt we will need this, chatgpt reccomendation
        """
        Simulate unaffected price path using Geometric Brownian Motion:
            S_k = S_{k-1} * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z_k)

        Returns
        -------
        S : np.ndarray
            Unaffected price path of length N+1
        """
        if seed is not None:
            np.random.seed(seed)

        S = np.zeros(self.N + 1)
        S[0] = self.S0

        for k in range(1, self.N + 1):
            Zk = np.random.randn()
            S[k] = S[k - 1] * np.exp((mu - 0.5 * self.sigma**2) * self.dt
                                     + self.sigma * self.sqrt_dt * Zk)

        return S

    def apply_market_impact(self, P, trades):
        """
        Apply permanent and temporary impact.

        Permanent impact:
            execution base price at step k uses
            P_k^perm = P_k - gamma * cumulative_shares_sold_before_k

        Temporary impact:
            execution price:
            P_k^exec = P_k^perm - eta * (n_k / dt)

        Parameters
        ----------
        P : np.ndarray
            Unaffected price path of length N+1
        trades : np.ndarray
            Shares traded each interval, length N

        Returns
        -------
        result : dict
            Dictionary containing:
            - permanent_prices
            - execution_prices
            - cashflows
            - total_cash
        """
        trades = np.asarray(trades)

        if len(trades) != self.N:
            raise ValueError("trades must have length N")

        permanent_prices = np.zeros(self.N)
        execution_prices = np.zeros(self.N)
        cashflows = np.zeros(self.N)

        cumulative_sold = 0.0

        for k in range(self.N):
            # Use unaffected price at start of interval k
            base_price = P[k]

            # Permanent impact from shares already sold
            permanent_prices[k] = base_price - self.gamma * cumulative_sold

            # Temporary impact from current trading rate
            execution_prices[k] = permanent_prices[k] - self.eta * (trades[k] / self.dt)

            # Cash received from selling trades[k] shares
            cashflows[k] = execution_prices[k] * trades[k]

            # Update cumulative sold after this trade
            cumulative_sold += trades[k]

        total_cash = np.sum(cashflows)

        return {
            "permanent_prices": permanent_prices,
            "execution_prices": execution_prices,
            "cashflows": cashflows,
            "total_cash": total_cash
        }

    def implementation_shortfall(self, X, total_cash):
        """
        Compute implementation shortfall:
            IS = initial paper value - realized cash
               = X * S0 - total_cash

        Parameters
        ----------
        X : float
            Total initial shares
        total_cash : float
            Total realized cash from execution

        Returns
        -------
        float
            Implementation shortfall
        """
        return X * self.S0 - total_cash


class Backtester:
    def __init__(self, strategy_model, market_env):
        self.strategy = strategy_model
        self.market = market_env

    def run(self, seed=None, forced_liquidation_discount=0.05): # not necesarily sure if seed is needed
        """
        Run the backtest.

        Parameters
        ----------
        seed : int or None
            Random seed for unaffected price path
        forced_liquidation_discount : float
            Extra percentage discount applied to final forced liquidation
            if inventory remains at T

        Returns
        -------
        log_df : pandas.DataFrame
            Transaction log
        summary : dict
            Summary statistics
        """
        # Strategy outputs
        times = self.strategy.times
        inventory_path = self.strategy.compute_inventory_trajectory()
        trades = self.strategy.compute_trade_list()

        # Market path
        prices = self.market.simulate_unaffected_price_abm(seed=seed)

        rows = []
        cumulative_sold = 0.0
        cumulative_cash = 0.0

        for k in range(self.strategy.N):
            t = times[k]
            x_before = inventory_path[k]
            n_k = trades[k]

            # Unaffected price at start of interval
            Pk = prices[k]

            # Permanent impact from previously sold inventory
            perm_price = Pk - self.market.gamma * cumulative_sold

            # Temporary impact from current trade rate
            exec_price = perm_price - self.market.eta * (n_k / self.market.dt)

            cash_captured = n_k * exec_price
            cumulative_cash += cash_captured
            cumulative_sold += n_k

            x_after = x_before - n_k

            rows.append({
                "step": k,
                "time": t,
                "unaffected_price": Pk,
                "inventory_before": x_before,
                "shares_traded": n_k,
                "inventory_after": x_after,
                "permanent_impact_price": perm_price,
                "execution_price": exec_price,
                "cash_captured": cash_captured,
                "cumulative_cash": cumulative_cash
            })

        # Final inventory check
        final_inventory = inventory_path[-1]
        penalty_cash = 0.0
        penalty_price = np.nan

        if final_inventory > 1e-8:
            # Liquidate remaining inventory at a steep penalty
            final_market_price = prices[-1] - self.market.gamma * cumulative_sold
            penalty_price = final_market_price * (1.0 - forced_liquidation_discount)
            penalty_cash = final_inventory * penalty_price
            cumulative_cash += penalty_cash

            rows.append({
                "step": self.strategy.N,
                "time": self.strategy.T,
                "unaffected_price": prices[-1],
                "inventory_before": final_inventory,
                "shares_traded": final_inventory,
                "inventory_after": 0.0,
                "permanent_impact_price": final_market_price,
                "execution_price": penalty_price,
                "cash_captured": penalty_cash,
                "cumulative_cash": cumulative_cash
            })

        log_df = pd.DataFrame(rows)

        initial_value = self.strategy.X * self.market.S0
        implementation_shortfall = initial_value - cumulative_cash

        summary = {
            "initial_portfolio_value": initial_value,
            "total_cash_received": cumulative_cash,
            "implementation_shortfall": implementation_shortfall,
            "final_inventory": max(0.0, final_inventory),
            "penalty_cash": penalty_cash,
            "penalty_price": penalty_price
        }

        return log_df, summary

class MonteCarloSimulator:
    def __init__(self, S0, X, T, N, sigma, eta, gamma):
        self.S0 = S0
        self.X = X
        self.T = T
        self.N = N
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma

    def run_single_lambda(self, lambd, n_sims=1000, seed=None): #very likely this function will be too slow and not needed
        """
        Run Monte Carlo for one lambda value.
        Returns implementation shortfall samples and summary stats.
        """
        rng = np.random.default_rng(seed)

        strategy = AlmgrenChrissModel(
            X=self.X,
            T=self.T,
            N=self.N,
            sigma=self.sigma,
            lambd=lambd,
            eta=self.eta,
            gamma=self.gamma
        )

        market = MarketEnvironment(
            S0=self.S0,
            sigma=self.sigma,
            T=self.T,
            N=self.N,
            gamma=self.gamma,
            eta=self.eta
        )

        trades = strategy.compute_trade_list()
        inventory = strategy.compute_inventory_trajectory()

        is_samples = np.zeros(n_sims)

        for i in range(n_sims):
            price_path = market.simulate_unaffected_price_abm(rng=rng)
            _, total_cash = market.execute_trades(price_path, trades)
            is_samples[i] = market.implementation_shortfall(self.X, total_cash)

        result = {
            "lambda": lambd,
            "kappa": strategy.compute_kappa(),
            "trade_list": trades,
            "inventory_path": inventory,
            "is_samples": is_samples,
            "mean_is": np.mean(is_samples),
            "var_is": np.var(is_samples, ddof=1),
            "std_is": np.std(is_samples, ddof=1)
        }

        return result

    def run_lambda_grid(self, lambda_values, n_sims=1000, seed=None): #likelhy same here
        """
        Run Monte Carlo across many lambda values.
        Returns DataFrame with mean and variance of IS for each lambda.
        """
        results = []

        for j, lambd in enumerate(lambda_values):
            res = self.run_single_lambda(
                lambd=lambd,
                n_sims=n_sims,
                seed=None if seed is None else seed + j
            )

            results.append({
                "lambda": lambd,
                "mean_is": res["mean_is"],
                "var_is": res["var_is"],
                "std_is": res["std_is"],
                "kappa": res["kappa"]
            })

        return pd.DataFrame(results)