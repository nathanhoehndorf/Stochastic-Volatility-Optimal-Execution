import numpy as np

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

    def simulate_unaffected_price_gbm(self, mu=0.0, seed=None):
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



