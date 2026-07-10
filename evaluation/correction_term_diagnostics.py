import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.AlmgrenChrissModel import AlmgrenChrissModel
from core.MarketEnvironment import MarketEnvironment


def _build_model_and_market(base, hparams):
    model = AlmgrenChrissModel(
        X=base["X"],
        T=base["T"],
        N=base["N"],
        sigma=base["sigma"],
        lambd=base["lambd"],
        eta=base["eta"],
        gamma=base["gamma"],
        xi=hparams["xi"],
        rho=hparams["rho"],
        v0=hparams["v0"],
        theta=hparams["theta"],
        omega=hparams["omega"],
    )

    market = MarketEnvironment(
        S0=base["S0"],
        sigma=base["sigma"],
        T=base["T"],
        N=base["N"],
        gamma=base["gamma"],
        eta=base["eta"],
        heston_params=hparams,
    )

    return model, market


def run_scenario_sensitivity(base, scenarios, n_sims=300):
    print("\n=== Scenario Sensitivity ===")
    for name, hparams in scenarios.items():
        model, market = _build_model_and_market(base, hparams)
        ac_trades = model.compute_trade_list(use_correction=False)

        _, var_path = market.simulate_unaffected_price_heston(seed=1, **hparams)
        corr_trades = model.compute_trade_list(use_correction=True, variance_path=var_path)
        delta = corr_trades - ac_trades

        diffs = []
        for seed in range(n_sims):
            price_path, variance_path = market.simulate_unaffected_price_heston(
                seed=seed,
                **hparams,
            )
            corr = model.compute_trade_list(use_correction=True, variance_path=variance_path)
            cash_ac = market.apply_market_impact(price_path, ac_trades)["total_cash"]
            cash_h = market.apply_market_impact(price_path, corr)["total_cash"]
            diffs.append(
                market.implementation_shortfall(base["X"], cash_ac)
                - market.implementation_shortfall(base["X"], cash_h)
            )

        print(f"\n[{name}]")
        print(f"hparams={hparams}")
        print(f"delta_trade max abs={np.max(np.abs(delta)):.9f}")
        print(f"delta_trade mean abs={np.mean(np.abs(delta)):.9f}")
        print(f"AC-Heston IS mean={np.mean(diffs):.9f}")
        print(f"AC-Heston IS mean abs={np.mean(np.abs(diffs)):.9f}")


def run_inventory_scaling_probe(base, hparams, seed=7):
    print("\n=== Inventory Scaling Probe ===")
    model, market = _build_model_and_market(base, hparams)
    _, variance_path = market.simulate_unaffected_price_heston(seed=seed, **hparams)

    ac_x = model.compute_inventory_trajectory()
    ac_trades = model.compute_trade_list(use_correction=False)
    base_chunks = model._compute_trade_correction_chunks(variance_path)

    # A: current implementation
    trades_a = model.compute_trade_list(use_correction=True, variance_path=variance_path)

    # B: multiply by inventory fraction x_t / X
    chunks_b = base_chunks * (ac_x[:-1] / model.X)
    trades_b = np.maximum(ac_trades + chunks_b, 0.0)
    trades_b *= model.X / np.sum(trades_b)

    # C: multiply by raw inventory x_t
    chunks_c = base_chunks * ac_x[:-1]
    trades_c = np.maximum(ac_trades + chunks_c, 0.0)
    trades_c *= model.X / np.sum(trades_c)

    print(f"b(t) max={np.max(model.compute_b_trajectory()):.9f}")
    print(f"base chunk max abs={np.max(np.abs(base_chunks)):.9f}")

    cash_ac = market.apply_market_impact(
        market.simulate_unaffected_price_heston(seed=seed, **hparams)[0],
        ac_trades,
    )["total_cash"]

    for label, trades in [
        ("A-current", trades_a),
        ("B-xfrac", trades_b),
        ("C-xraw", trades_c),
    ]:
        price_path, _ = market.simulate_unaffected_price_heston(seed=seed, **hparams)
        cash_h = market.apply_market_impact(price_path, trades)["total_cash"]
        diff = market.implementation_shortfall(base["X"], cash_ac) - market.implementation_shortfall(
            base["X"],
            cash_h,
        )
        print(
            f"{label}: max delta abs={np.max(np.abs(trades - ac_trades)):.9f}, "
            f"AC-Heston diff={diff:.9f}, "
            f"delta sumsq={np.sum(trades**2)-np.sum(ac_trades**2):.9f}"
        )


def main():
    base = {
        "S0": 100.0,
        "X": 10000.0,
        "T": 1.0,
        "N": 78,
        "sigma": 0.08,
        "lambd": 0.5,
        "eta": 0.01,
        "gamma": 1e-6,
    }

    scenarios = {
        "equilibrium": {
            "v0": 0.04,
            "mu": 0.0,
            "theta": 2.0,
            "omega": 0.04,
            "xi": 0.2,
            "rho": -0.4,
        },
        "shock-1": {
            "v0": 0.16,
            "mu": 0.0,
            "theta": 2.0,
            "omega": 0.04,
            "xi": 0.2,
            "rho": -0.4,
        },
        "shock-2": {
            "v0": 0.12,
            "mu": 0.0,
            "theta": 2.0,
            "omega": 0.03,
            "xi": 0.2,
            "rho": -0.4,
        },
    }

    run_scenario_sensitivity(base, scenarios, n_sims=300)
    run_inventory_scaling_probe(base, scenarios["shock-2"], seed=7)


if __name__ == "__main__":
    main()
