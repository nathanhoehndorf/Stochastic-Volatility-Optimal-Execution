"""
Synthetic control sweep for the Heston correction term.

Purpose: Test the correction in a clean environment that strictly follows the model's
assumptions, free from LOBSTER microstructure calibration noise.  We sweep across
(rho, xi, v0/omega) combinations to map where the correction actually produces a
statistically significant execution improvement.

Run from the project root:
    python evaluation/synthetic_control_sweep.py
"""
import os
import sys
import itertools

import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.AlmgrenChrissModel import AlmgrenChrissModel
from core.MarketEnvironment import MarketEnvironment

# ─────────────────────────────────────────────────────────────────────────────
# Base execution parameters
# ─────────────────────────────────────────────────────────────────────────────
BASE = {
    "S0"   : 100.0,
    "X"    : 10_000.0,
    "T"    : 1.0,
    "N"    : 78,        # one 5-min interval per step, T=1 day convention
    "sigma": 0.08,      # daily vol consistent with omega = 0.0064 at equilibrium
    "lambd": 0.5,
    "eta"  : 0.01,
    "gamma": 1e-6,
}

# ─────────────────────────────────────────────────────────────────────────────
# Sweep grid: parameters that determine how large the correction can be
# ─────────────────────────────────────────────────────────────────────────────
# rho   : leverage effect (negative = equity-like)
# xi    : vol-of-vol (higher = more stochastic variance, larger correction)
# shock : v0 / omega ratio (how far above long-run mean we start)
# N_SIMS per scenario
RHO_VALS   = [-0.7, -0.5, -0.3, 0.0]
XI_VALS    = [0.1, 0.3, 0.6]
SHOCK_VALS = [1.0, 2.0, 4.0]   # v0 = shock * omega
OMEGA      = 0.04               # long-run daily variance (vol ≈ 20%)
THETA      = 2.0                # moderate mean-reversion speed
N_SIMS     = 1_000
SEED_BASE  = 42


def run_scenario(hparams, base, n_sims):
    """
    Run n_sims paired simulations for one Heston parameter set.
    Returns (is_diffs, sumsq_diffs, delta_trades_first_path).
    """
    model = AlmgrenChrissModel(
        X     = base["X"],
        T     = base["T"],
        N     = base["N"],
        sigma = base["sigma"],
        lambd = base["lambd"],
        eta   = base["eta"],
        gamma = base["gamma"],
        xi    = hparams["xi"],
        rho   = hparams["rho"],
        v0    = hparams["v0"],
        theta = hparams["theta"],
        omega = hparams["omega"],
    )
    market = MarketEnvironment(
        S0          = base["S0"],
        sigma       = base["sigma"],
        T           = base["T"],
        N           = base["N"],
        gamma       = base["gamma"],
        eta         = base["eta"],
        heston_params = hparams,
    )

    classic_trades = model.compute_trade_list(use_correction=False)

    is_diffs        = np.zeros(n_sims)
    sumsq_diffs     = np.zeros(n_sims)
    delta_first     = None

    for i in range(n_sims):
        seed = SEED_BASE + i
        price_path, variance_path = market.simulate_unaffected_price_heston(
            seed=seed, **hparams
        )

        heston_trades = model.compute_trade_list(
            use_correction=True, variance_path=variance_path
        )

        if delta_first is None:
            delta_first = heston_trades - classic_trades

        cash_ac   = market.apply_market_impact(price_path, classic_trades)["total_cash"]
        cash_hest = market.apply_market_impact(price_path, heston_trades)["total_cash"]

        is_ac   = market.implementation_shortfall(base["X"], cash_ac)
        is_hest = market.implementation_shortfall(base["X"], cash_hest)

        is_diffs[i]    = is_ac - is_hest          # positive → Heston wins
        sumsq_diffs[i] = np.sum(heston_trades**2) - np.sum(classic_trades**2)

    return is_diffs, sumsq_diffs, delta_first


def stats_for_diffs(diffs):
    """Return (mean, std, t, p_one_sided, wilcoxon_p)."""
    mean = float(np.mean(diffs))
    std  = float(np.std(diffs, ddof=1))

    nz = diffs[diffs != 0]
    if len(nz) > 1 and not np.allclose(diffs, 0):
        t, p_two = ttest_1samp(diffs, 0.0)
        p_one = p_two / 2 if t > 0 else 1.0 - p_two / 2
        _, p_w = wilcoxon(diffs, alternative="greater")
    else:
        # all differences are exactly zero — correction produced no change
        t, p_one, p_w = 0.0, 0.5, None

    return mean, std, float(t), float(p_one), p_w


def main():
    sep  = "─" * 100
    wide = "═" * 100

    print(f"\n{wide}")
    print("  SYNTHETIC CONTROL SWEEP  —  Heston correction vs AC baseline")
    print(f"  n_sims per scenario = {N_SIMS}  |  omega = {OMEGA}  theta = {THETA}")
    print(f"{wide}\n")

    print(f"{'rho':>6} {'xi':>6} {'shock':>7} {'v0':>7} | "
          f"{'mean_diff':>11} {'t':>8} {'p_1s':>8} {'p_wil':>8} | "
          f"{'max|Δn|':>10} {'mean|Δn|':>10} {'Heston wins':>12}")
    print(sep)

    summary_rows = []

    for rho, xi, shock in itertools.product(RHO_VALS, XI_VALS, SHOCK_VALS):
        v0 = shock * OMEGA

        # Feller check: 2*theta*omega >= xi^2 ; if violated, skip (model degenerate)
        if 2 * THETA * OMEGA <= xi ** 2:
            print(f"{rho:>6.1f} {xi:>6.2f} {shock:>7.1f} {v0:>7.4f} | "
                  f"  [Feller violated — skipped]")
            continue

        hparams = {
            "v0"   : v0,
            "mu"   : 0.0,
            "theta": THETA,
            "omega": OMEGA,
            "xi"   : xi,
            "rho"  : rho,
        }

        is_diffs, sumsq_diffs, delta_first = run_scenario(hparams, BASE, N_SIMS)

        mean, std, t, p_one, p_w = stats_for_diffs(is_diffs)
        max_delta  = float(np.max(np.abs(delta_first)))
        mean_delta = float(np.mean(np.abs(delta_first)))
        wins       = mean > 0 and p_one < 0.05

        p_w_str = f"{p_w:.4f}" if p_w is not None else "  N/A  "

        print(f"{rho:>6.1f} {xi:>6.2f} {shock:>7.1f} {v0:>7.4f} | "
              f"{mean:>11.4f} {t:>8.4f} {p_one:>8.4f} {p_w_str:>8} | "
              f"{max_delta:>10.4f} {mean_delta:>10.4f} {'✓' if wins else '':>12}")

        summary_rows.append({
            "rho": rho, "xi": xi, "shock": shock, "v0": v0,
            "mean_diff": mean, "t": t, "p_one": p_one, "p_wilcoxon": p_w,
            "max_delta": max_delta, "mean_delta": mean_delta, "wins": wins,
        })

    print(sep)
    winners = [r for r in summary_rows if r["wins"]]
    print(f"\nScenarios where Heston is statistically better (p<0.05, mean_diff>0): "
          f"{len(winners)} / {len(summary_rows)}")

    if winners:
        print("\nTop scenarios by mean IS improvement:")
        for r in sorted(winners, key=lambda x: -x["mean_diff"])[:5]:
            print(f"  rho={r['rho']:+.1f}  xi={r['xi']:.2f}  shock={r['shock']:.1f}x  "
                  f"mean_diff={r['mean_diff']:.4f}  t={r['t']:.3f}  p={r['p_one']:.4f}")

    # ── sensitivity summary by parameter ─────────────────────────────────────
    print(f"\n{sep}")
    print("MEAN IS IMPROVEMENT BY PARAMETER (averaged across all other parameters)")
    print(sep)

    for label, key, vals in [("rho", "rho", RHO_VALS), ("xi", "xi", XI_VALS),
                              ("shock", "shock", SHOCK_VALS)]:
        means = []
        for v in vals:
            group = [r["mean_diff"] for r in summary_rows if r[key] == v]
            means.append((v, float(np.mean(group)) if group else float("nan")))
        row_str = "  ".join(f"{key}={v:+.2f} → {m:+.4f}" for v, m in means)
        print(f"  {label}: {row_str}")

    print(f"\n{wide}\n")


if __name__ == "__main__":
    main()
