import numpy as np
import pandas as pd

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
