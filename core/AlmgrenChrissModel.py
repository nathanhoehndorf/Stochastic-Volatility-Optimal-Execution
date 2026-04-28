import numpy as np

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

        tau = self.T - self.times
        T = self.T

        if np.any(kappa * T > 700):
            log_ratio = -kappa * (T-tau)
            adjustment = (1-np.exp(-2*kappa*tau))/(1-np.exp(-2*kappa*T))
            return self.X * np.exp(log_ratio) * adjustment
        else:
            return self.X * np.sinh(kappa*tau)/np.sinh(kappa*T)

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
