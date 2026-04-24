import AlmgrenChrissModel as ac
import MarketEnvironment as me
import numpy as np
import pandas as pd

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

        strategy = ac.AlmgrenChrissModel(
            X=self.X,
            T=self.T,
            N=self.N,
            sigma=self.sigma,
            lambd=lambd,
            eta=self.eta,
            gamma=self.gamma
        )

        market = me.MarketEnvironment(
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
            price_path = market.simulate_unaffected_price_abm(seed=seed)
            total_cash = market.apply_market_impact(price_path, trades) # make new function? Find other name in MarketEnvironment? 
            total_cash = total_cash['total_cash']
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