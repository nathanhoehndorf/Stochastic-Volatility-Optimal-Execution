import numpy as np
import pandas as pd

class MarketEnvironment:
    def simulate_heston_paths_vectorized(
        self,
        n_sims,
        v0,
        mu=0.0,
        theta=2.0,
        omega=0.04,
        xi=0.3,
        rho=-0.7,
        seed=None
    ):
        """
        Generate many Heston price paths and variance paths at once.

        Returns
        -------
        S : np.ndarray, shape (n_sims, N + 1)
            Simulated price paths. Each row is one price path.

        v : np.ndarray, shape (n_sims, N + 1)
            Simulated variance paths. Each row is one variance path.
        """
        rng = np.random.default_rng(seed)

        S = np.zeros((n_sims, self.N + 1))
        v = np.zeros((n_sims, self.N + 1))

        S[:, 0] = self.S0
        v[:, 0] = v0

        corr_matrix = np.array([
            [1.0, rho],
            [rho, 1.0]
        ])

        L = np.linalg.cholesky(corr_matrix)

        # Shape: (n_sims, N, 2)
        z = rng.standard_normal((n_sims, self.N, 2))

        # Apply Cholesky correlation to every pair of shocks
        z_corr = z @ L.T

        Z1 = z_corr[:, :, 0]  # variance shocks
        Z2 = z_corr[:, :, 1]  # price shocks

        for k in range(self.N):
            v_safe = np.maximum(v[:, k], 0.0)

            v[:, k + 1] = (
                v[:, k]
                + theta * (omega - v_safe) * self.dt
                + xi * np.sqrt(v_safe * self.dt) * Z1[:, k]
            )

            v[:, k + 1] = np.maximum(v[:, k + 1], 0.0)

            S[:, k + 1] = (
                S[:, k]
                + mu * S[:, k] * self.dt
                + np.sqrt(v_safe * self.dt) * S[:, k] * Z2[:, k]
            )

        return S, v
    def simulate_unaffected_price_heston(
        self,
        v0,
        mu=0.0,
        theta=2.0,
        omega=0.04,
        xi=0.3,
        rho=-0.7,
        seed=None
    ):
        """
    Simulate an asset price path and its stochastic variance using the Heston model
    with an Euler-Maruyama discretization scheme.

    The continuous-time dynamics are:

        dS_t = mu * S_t dt + sqrt(v_t) * S_t dW_2
        dv_t = theta * (omega - v_t) dt + xi * sqrt(v_t) dW_1

    where:
        - S_t is the asset price
        - v_t is the instantaneous variance
        - mu is the drift of the price process
        - theta is the mean-reversion speed of variance
        - omega is the long-run (mean) variance
        - xi is the volatility of volatility ("vol of vol")
        - dW_1 and dW_2 are Brownian motions with correlation rho

    The correlation between dW_1 and dW_2 is enforced via a Cholesky
    decomposition of the covariance matrix.

    Parameters
    ----------
    v0 : float
        Initial variance (must be non-negative)

    mu : float, optional
        Drift of the asset price (default: 0.0)

    theta : float, optional
        Mean-reversion speed of the variance process

    omega : float, optional
        Long-run average variance level

    xi : float, optional
        Volatility of volatility (controls randomness of variance)

    rho : float, optional
        Correlation between price and variance shocks (-1 <= rho <= 1)

    seed : int or None, optional
        Random seed for reproducibility

    Returns
    -------
    S : np.ndarray of shape (N + 1,)
        Simulated asset price path, including initial price S0

    v : np.ndarray of shape (N + 1,)
        Simulated variance path, including initial variance v0

    Notes
    -----
    - Uses Euler-Maruyama discretization:
          v_{t+dt} = v_t + theta(omega - v_t)dt + xi sqrt(v_t dt) * Z1
          S_{t+dt} = S_t + mu S_t dt + sqrt(v_t dt) S_t * Z2

    - Variance is truncated at zero to prevent numerical instability:
          v_t = max(v_t, 0)

    - Z1 and Z2 are correlated standard normal variables constructed via:
          [Z1, Z2]^T = L * [z1, z2]^T
      where L is the Cholesky factor of the correlation matrix.

    - This function generates a *single* path. For Monte Carlo,
      call it multiple times or vectorize.
    """
        rng = np.random.default_rng(seed)

        S = np.zeros(self.N + 1)
        v = np.zeros(self.N + 1)

        S[0] = self.S0
        v[0] = v0

        corr = np.array([
            [1.0, rho],
            [rho, 1.0]
        ])

        L = np.linalg.cholesky(corr)

        for k in range(self.N):
            z_independent = rng.standard_normal(2)
            z_corr = L @ z_independent

            Z1 = z_corr[0]  # variance noise
            Z2 = z_corr[1]  # price noise

            v_safe = max(v[k], 0.0)

            v[k + 1] = (
                v[k]
                + theta * (omega - v_safe) * self.dt
                + xi * np.sqrt(v_safe * self.dt) * Z1
            )

            v[k + 1] = max(v[k + 1], 0.0)

            S[k + 1] = (
                S[k]
                + mu * S[k] * self.dt
                + np.sqrt(v_safe * self.dt) * S[k] * Z2
            )

        return S, v
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

    def simulate_unaffected_price_gbm(self, mu=0.0, seed=None): # I highly doubt we will need this, chatgpt reccomendation - Ben;;; I agree, the model uses ABM for the unaffected price.
                                                                                                                                ### However, GBM is more realistic for stock prices, so we can use this option
                                                                                                                                ### if we wanted to do further testing and change another variable.
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
        return X * self.S0 - int(total_cash)
