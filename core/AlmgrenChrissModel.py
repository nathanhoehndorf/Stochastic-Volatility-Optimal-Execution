import numpy as np

class AlmgrenChrissModel:
    def __init__(self, X, T, N, sigma, lambd, eta, gamma, xi=None, rho=None):
        """
        Core Almgren-Chriss execution model with optional Heston asymptotic correction.

        Parameters
        ----------
        X : float
            Total number of shares to execute
        T : float
            Total trading horizon
        N : int
            Number of trading intervals
        sigma : float
            Asset volatility (used as sqrt(v0) for Heston)
        lambd : float
            Risk aversion parameter
        eta : float
            Temporary market impact coefficient
        gamma : float
            Permanent market impact coefficient
        xi : float, optional
            Volatility of volatility for Heston dynamics
        rho : float, optional
            Correlation between price and variance for Heston dynamics
        """
        self.X = X
        self.T = T
        self.N = N
        self.sigma = sigma
        self.lambd = lambd
        self.eta = eta
        self.gamma = gamma
        self.xi = xi
        self.rho = rho

        self.dt = T / N
        self.times = np.linspace(0, T, N + 1)

    def compute_kappa(self):
        """
        Compute kappa = sqrt(lambda * sigma^2 / eta)
        """
        return np.sqrt((self.lambd * self.sigma**2) / self.eta)

    def compute_inventory_trajectory(self):
        """
        Compute optimal remaining shares x_t at each grid point (classic AC).
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
            # Handle tau=0 at the end
            res = np.zeros_like(tau)
            nonzero = tau > 0
            res[nonzero] = self.X * np.sinh(kappa*tau[nonzero])/np.sinh(kappa*T)
            res[~nonzero] = 0
            return res

    def compute_perturbed_inventory_trajectory(self):
        """
        Compute optimal remaining shares x_t using the Heston asymptotic correction.
        The rate formula n_t* = -dx/dt is integrated via Euler steps.
        """
        if self.xi is None or self.rho is None:
            return self.compute_inventory_trajectory()

        x = np.zeros(self.N + 1)
        x[0] = self.X
        kappa = self.compute_kappa()
        
        for k in range(self.N):
            tau = self.T - self.times[k]
            
            if tau <= 1e-10:
                x[k+1] = 0
                continue
            
            # AC Base Term: kappa * coth(kappa * tau) * x_t
            coth_kt = 1.0 / np.tanh(kappa * tau)
            term1 = kappa * coth_kt * x[k]
            
            # Heston Correction Term
            csch_kt = 1.0 / np.sinh(kappa * tau)
            A = np.tanh(kappa * tau / 2.0) * (coth_kt - kappa * tau * (csch_kt**2))
            correction = (self.xi * self.rho * x[k] / (4.0 * self.eta)) * A
            
            rate = term1 - correction
            x[k+1] = x[k] - rate * self.dt
            
            # Liquidation constraint
            x[k+1] = max(x[k+1], 0)
            
        return x

    def compute_trade_list(self, use_correction=False):
        """
        Compute shares traded each step:
        n_k = x_{k-1} - x_k

        Returns array of length N.
        """
        if use_correction and self.xi is not None and self.rho is not None:
            x = self.compute_perturbed_inventory_trajectory()
        else:
            x = self.compute_inventory_trajectory()
            
        n = x[:-1] - x[1:]
        return n

    def summary(self, use_correction=False):
        """
        Return all major outputs in a dictionary.
        """
        x = (self.compute_perturbed_inventory_trajectory() 
             if use_correction else self.compute_inventory_trajectory())
        n = x[:-1] - x[1:]
        kappa = self.compute_kappa()

        return {
            "kappa": kappa,
            "times": self.times,
            "inventory": x,
            "trades": n,
            "dt": self.dt,
            "is_perturbed": use_correction
        }
