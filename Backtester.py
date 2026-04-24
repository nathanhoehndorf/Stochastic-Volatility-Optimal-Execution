import pandas as pd
import numpy as np

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
        prices, variances = self.market.simulate_unaffected_price_heston(
        v0=0.04,
        mu=0.0,
        theta=2.0,
        omega=0.04,
        xi=0.3,
        rho=-0.7,
        seed=seed)

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
