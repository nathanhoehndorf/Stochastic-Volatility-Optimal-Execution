import pandas as pd
import numpy as np
from evaluation.statistics import calculate_test_suite
from core.MonteCarloSimulator import MonteCarloSimulator

class ModelComparator:
    def __init__(self, model_ac, model_hest, market_env, num_sims=1000, seed=None):
        """
        Parameters
        ----------
        model_ac    : AlmgrenChrissModel — provides the optimal trading strategy.
                      Expected attributes: X, T, N, sigma, eta, gamma, lambd, xi, rho
        model_hest  : HestonModel (or similar) — provides Heston parameters only.
                      Expected attributes: v0, mu, theta, omega, xi, rho
        market_env  : MarketEnvironment — used directly for price simulation and IS calc.
                      Expected attributes: S0
        num_sims    : number of Monte Carlo paths per model
        seed        : base RNG seed; both models draw from the same seed sequence so
                      results are directly comparable
        """
        self.model_ac = model_ac
        self.model_hest = model_hest
        self.market_env = market_env
        self.num_sims = num_sims
        self.seed = seed 
        
        # Sync Heston parameters to the AC model if provided
        if hasattr(model_hest, 'xi'):
            self.model_ac.xi = model_hest.xi
        if hasattr(model_hest, 'rho'):
            self.model_ac.rho = model_hest.rho

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_simulator(self):
        """Build a MonteCarloSimulator from the AC model's parameters."""
        m = self.model_ac
        return MonteCarloSimulator(
            S0    = self.market_env.S0,
            X     = m.X,
            T     = m.T,
            N     = m.N,
            sigma = m.sigma,
            eta   = m.eta,
            gamma = m.gamma,
        )
    
    def _simulate_heston_path(self, path_seed, path_index):
        """Generate one shared Heston path for paired strategy evaluation."""
        h = self.model_hest
        price_path, variance_path = self.market_env.simulate_unaffected_price_heston(
            v0=h.v0,
            mu=h.mu,
            theta=h.theta,
            omega=h.omega,
            xi=h.xi,
            rho=h.rho,
            seed=path_seed,
        )

        if not np.all(np.isfinite(variance_path)):
            print(f"Warning: invalid Heston variance path {path_index} contains non-finite values")
            return None, None, np.nan

        if not np.all(np.isfinite(price_path)):
            print(f"Warning: invalid Heston price path {path_index} contains non-finite values")
            return None, None, np.nan

        if np.nanmax(price_path) > self.market_env.S0 * 2.0 or np.nanmin(price_path) < self.market_env.S0 * 0.2:
            print(f"Warning: Heston price path {path_index} is implausible (extreme price move)")
            return None, None, np.nan

        path_vols = np.sqrt(np.maximum(variance_path, 0.0))
        regime_vol = float(np.mean(path_vols))
        return price_path, variance_path, regime_vol

    def _evaluate_trades_on_price_path(self, trades, price_path, label, path_index):
        """Apply one trade schedule to one shared price path."""
        try:
            total_cash = self.market_env.apply_market_impact(price_path, trades)
            is_val = self.market_env.implementation_shortfall(
                self.model_ac.X, total_cash["total_cash"]
            )
            if not np.isfinite(is_val):
                raise ValueError("non-finite implementation shortfall")
            return is_val
        except Exception as exc:
            print(f"Warning: invalid {label} path {path_index}: {exc}")
            return np.nan

    def _filter_valid_pairs(self, is_ac, is_hest, starting_vols):
        mask = np.isfinite(is_ac) & np.isfinite(is_hest)
        valid_count = int(mask.sum())
        if valid_count == 0:
            raise ValueError("No valid paired samples available after filtering invalid paths.")
        if valid_count < len(is_ac):
            print(f"Warning: filtered {len(is_ac) - valid_count} invalid paired samples.")
        if starting_vols is not None:
            start_vols = np.asarray(starting_vols, dtype=float)
            return is_ac[mask], is_hest[mask], start_vols[mask]
        return is_ac[mask], is_hest[mask], None

    def run_comparison(self, stat_kwargs=None):
        """
        Run Monte Carlo comparison:
        1. Classic AC strategy under Heston dynamics (Baseline)
        2. Heston-Corrected AC strategy under Heston dynamics (Challenger)
 
        Returns
        -------
        results : dict from calculate_test_suite, plus:
                  "is_ac"       — raw IS array under GBM dynamics (Classic)
                  "is_heston"   — raw IS array under Heston dynamics (Corrected)
                  "starting_vols" - per-path volatility metric for regime analysis
        """
        stat_kwargs = stat_kwargs or {}
        
        # 1. Baseline: Classic AC trades
        trades_classic = self.model_ac.compute_trade_list(use_correction=False)
        
        # 2. Challenger: Heston-Corrected trades
        trades_corrected = self.model_ac.compute_trade_list(use_correction=True)

        rng = np.random.default_rng(self.seed)
        is_ac = np.full(self.num_sims, np.nan)
        is_heston = np.full(self.num_sims, np.nan)
        starting_vols = np.full(self.num_sims, np.nan)

        print(f"Running {self.num_sims} paired Heston-path simulations ...")
        for i in range(self.num_sims):
            path_seed = int(rng.integers(0, 1_000_000_000))
            price_path, variance_path, regime_vol = self._simulate_heston_path(path_seed, i)

            if price_path is None:
                continue

            starting_vols[i] = regime_vol
            is_ac[i] = self._evaluate_trades_on_price_path(
                trades_classic, price_path, "AC", i
            )
            corrected_trades = self.model_ac.compute_trade_list(
                use_correction=True,
                variance_path=variance_path,
            )
            is_heston[i] = self._evaluate_trades_on_price_path(
                corrected_trades, price_path, "Heston", i
            )
 
        is_ac, is_heston, starting_vols = self._filter_valid_pairs(is_ac, is_heston, starting_vols)
 
        # Allow caller to override the regime metric (e.g. collected externally)
        starting_vols = stat_kwargs.pop("starting_vols", starting_vols)
 
        print("Running statistical test suite ...")
        results = calculate_test_suite(
            is_ac         = is_ac,
            is_hest       = is_heston,
            starting_vols = starting_vols,
            **stat_kwargs,
        )

        results["is_ac"]        = is_ac
        results["is_heston"]    = is_heston
        results["starting_vols"] = starting_vols
 
        return results
