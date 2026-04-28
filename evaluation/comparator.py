import pandas as pd
import numpy as np
from evaluation.statistics import calculate_test_suite
from MonteCarloSimulator import MonteCarloSimulator

class ModelComparator:
    def __init__(self, model_ac, model_hest, market_env, num_sims=1000, seed=None):
        """
        Parameters
        ----------
        model_ac    : AlmgrenChrissModel — provides the optimal trading strategy.
                      Expected attributes: X, T, N, sigma, eta, gamma, lambd
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
    
    def _run_ac_paths(self, trades, rng):
        """
        Evaluate the AC trading strategy on GBM price paths.
        Returns an IS array of length num_sims.
        """
        is_samples = np.zeros(self.num_sims)
        for i in range(self.num_sims):
            path_seed = int(rng.integers(0, 1_000_000_000))
            price_path = self.market_env.simulate_unaffected_price_abm(seed=path_seed)
            total_cash = self.market_env.apply_market_impact(price_path, trades)
            is_samples[i] = self.market_env.implementation_shortfall(
                self.model_ac.X, total_cash["total_cash"]
            )
        return is_samples
    
    def _run_heston_paths(self, trades, rng):
        """
        Evaluate the AC trading strategy on Heston price paths.
        Returns an IS array of length num_sims, plus starting vol for each path.
        """
        h = self.model_hest
        is_samples    = np.zeros(self.num_sims)
        starting_vols = np.zeros(self.num_sims)
        for i in range(self.num_sims):
            path_seed = int(rng.integers(0, 1_000_000_000))
            price_path, variance_path = self.market_env.simulate_unaffected_price_heston(
                v0    = h.v0,
                mu    = h.mu,
                theta = h.theta,
                omega = h.omega,
                xi    = h.xi,
                rho   = h.rho,
                seed  = path_seed,
            )
            total_cash = self.market_env.apply_market_impact(price_path, trades)
            is_samples[i]    = self.market_env.implementation_shortfall(
                self.model_ac.X, total_cash["total_cash"]
            )
            starting_vols[i] = np.sqrt(variance_path[0])   # convert variance → vol
        return is_samples, starting_vols

    def run_comparison(self, stat_kwargs=None):
        """
        Run Monte Carlo under both GBM (AC) and Heston price dynamics using the
        same AC optimal trading strategy, then pass the paired IS arrays to
        calculate_test_suite.
 
        The two RNGs are seeded independently from the same base seed, so the
        paths are paired in sample size but not artificially correlated.
 
        Parameters
        ----------
        stat_kwargs : dict, optional
            Extra keyword arguments forwarded to calculate_test_suite
            (e.g. alpha_levels, n_bootstrap, n_regimes, rho_sweep).
 
        Returns
        -------
        results : dict from calculate_test_suite, plus:
                  "is_ac"       — raw IS array under GBM dynamics
                  "is_heston"   — raw IS array under Heston dynamics
                  "starting_vols" — initial vol for each Heston path (for regime analysis)
        """
        stat_kwargs = stat_kwargs or {}
        # Pre-compute the AC optimal strategy once — identical for both evaluations
        trades = self.model_ac.compute_trade_list()

        # Seed the two RNGs from the same base so the experiment is reproducible
        # but the two path sequences are independent
        rng_ac   = np.random.default_rng(self.seed)
        rng_hest = np.random.default_rng(
            None if self.seed is None else self.seed + 1
        )

        print(f"Running {self.num_sims} AC (GBM) paths ...")
        is_ac = self._run_ac_paths(trades, rng_ac)
 
        print(f"Running {self.num_sims} Heston paths ...")
        is_heston, starting_vols = self._run_heston_paths(trades, rng_hest)
 
        # Allow caller to override starting_vols (e.g. collected externally)
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