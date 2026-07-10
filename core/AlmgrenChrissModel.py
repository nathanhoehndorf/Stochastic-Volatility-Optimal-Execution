import numpy as np
from scipy.integrate import quad

class AlmgrenChrissModel:
    def __init__(self, X, T, N, sigma, lambd, eta, gamma, xi=None, rho=None, v0=None, theta=None, omega=None):
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
        v0 : float, optional
            Initial variance for deterministic Heston variance approximation
        theta : float, optional
            Mean reversion speed for deterministic Heston variance approximation
        omega : float, optional
            Long-run variance for deterministic Heston variance approximation
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
        self.v0 = sigma**2 if v0 is None else v0
        self.theta = theta
        self.omega = self.v0 if omega is None else omega

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

    def deterministic_variance(self, t):
        """Approximate the variance path by the deterministic Heston mean path."""
        if self.theta is None:
            return float(self.v0)
        return float(self.omega + (self.v0 - self.omega) * np.exp(-self.theta * t))

    def compute_b_value(self, t):
        """Compute the deterministic quadrature term b(t)."""
        tau = self.T - t

        if tau <= 1e-10 or self.xi is None or abs(self.xi) <= 1e-12 or self.theta is None:
            return 0.0

        theta = self.theta
        kappa = self.compute_kappa()
        if abs(kappa) <= 1e-12:
            return 0.0

        denom = np.sinh(kappa * tau)
        if abs(denom) <= 1e-12:
            return 0.0

        def integrand(s):
            if kappa < 1e-4:
                ratio = (self.T-s) / tau
            else:
                numerator = np.sinh(kappa * (self.T - s))
                ratio = numerator / denom
            return np.exp(-theta * (s - t)) * (ratio ** 2)

        integral_value, _ = quad(integrand, t, self.T, limit=100)
        return float((self.lambd / self.xi) * integral_value)

    def compute_b_trajectory(self):
        """Compute deterministic b(t) on the model time grid."""
        return np.array([self.compute_b_value(t) for t in self.times], dtype=float)

    def _resolve_variance_path(self, variance_path=None):
        """Return a variance path aligned to the model grid."""
        if variance_path is None:
            return np.array(
                [self.deterministic_variance(t) for t in self.times],
                dtype=float,
            )

        variance_path = np.asarray(variance_path, dtype=float)
        if variance_path.shape[0] != self.N + 1:
            raise ValueError("variance_path must have length N + 1")
        return variance_path

    def _compute_trade_correction_chunks(self, variance_path):
        """Compute per-step correction shares (rate correction multiplied by dt)."""
        correction_chunks = np.zeros(self.N, dtype=float)

        if self.xi is None or self.rho is None:
            return correction_chunks

        kappa = self.compute_kappa()
        if abs(kappa) <= 1e-12:
            return correction_chunks

        b_values = self.compute_b_trajectory()

        for k in range(self.N):
            tau = self.T - self.times[k]
            if tau <= 1e-10:
                continue

            v_t = float(max(variance_path[k], 0.0))
            h_v = self.xi * b_values[k]

            corr_rate = (
                self.xi * self.rho * v_t * h_v / (2.0 * self.eta * kappa)
            ) * np.tanh(kappa * tau / 2.0)

            correction_chunks[k] = corr_rate * self.dt

        return correction_chunks

    def compute_perturbed_inventory_trajectory(self, variance_path=None):
        """
        Compute optimal remaining shares x_t using the Heston asymptotic correction.
        The corrected trades are built as perturbations on exact AC trades.
        """
        if self.xi is None or self.rho is None:
            return self.compute_inventory_trajectory()

        trades = self.compute_trade_list(use_correction=True, variance_path=variance_path)
        x = np.zeros(self.N + 1, dtype=float)
        x[0] = self.X
        x[1:] = self.X - np.cumsum(trades)
        x = np.maximum(x, 0.0)
        x[-1] = 0.0
        return x

    def compute_trade_list(self, use_correction=False, variance_path=None):
        """
        Compute shares traded each step:
        n_k = x_{k-1} - x_k

        Returns array of length N.
        """
        x_classic = self.compute_inventory_trajectory()
        classic_trades = x_classic[:-1] - x_classic[1:]

        if not use_correction or self.xi is None or self.rho is None:
            return classic_trades

        variance_values = self._resolve_variance_path(variance_path=variance_path)
        correction_chunks = self._compute_trade_correction_chunks(variance_values)

        # Build corrected trades with dynamic inventory scaling at each step.
        corrected_trades = np.zeros(self.N, dtype=float)
        current_x = float(self.X)

        for k in range(self.N):
            if current_x <= 1e-12:
                corrected_trades[k] = 0.0
                continue

            # correction_chunks[k] is a per-step correction in shares per unit inventory.
            discrete_correction = correction_chunks[k] * current_x
            proposed_trade = classic_trades[k] + discrete_correction

            # Keep trades feasible and inventory non-negative.
            proposed_trade = min(max(proposed_trade, 0.0), current_x)
            corrected_trades[k] = proposed_trade
            current_x -= proposed_trade

        total_sold = float(np.sum(corrected_trades))
        if total_sold <= 1e-12:
            return classic_trades

        corrected_trades *= self.X / total_sold
        return corrected_trades

    def summary(self, use_correction=False, variance_path=None):
        """
        Return all major outputs in a dictionary.
        """
        x = (self.compute_perturbed_inventory_trajectory(variance_path=variance_path) 
             if use_correction else self.compute_inventory_trajectory())
        n = x[:-1] - x[1:]
        kappa = self.compute_kappa()

        return {
            "kappa": kappa,
            "times": self.times,
            "inventory": x,
            "trades": n,
            "b_values": self.compute_b_trajectory() if use_correction else None,
            "dt": self.dt,
            "is_perturbed": use_correction
        }
