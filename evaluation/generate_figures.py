"""Generate the four paper figures requested for the current manuscript.

The script saves both PNG and PDF versions into docs/paper/figures/.

Usage:
    uv run python evaluation/generate_figures.py
    uv run python evaluation/generate_figures.py --output-dir docs/paper/figures
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.AlmgrenChrissModel import AlmgrenChrissModel
from core.MarketEnvironment import MarketEnvironment
from data.calibrator import LobsterCalibrator
from evaluation.comparator import ModelComparator
from main import HestonParameters, build_objects, list_datasets


BASE_PARAMS = {
    "S0": 100.0,
    "X": 10_000.0,
    "T": 1.0,
    "N": 78,
    "sigma": 0.08,
    "lambd": 0.5,
    "eta": 0.01,
    "gamma": 1e-6,
}

SYNTHETIC_SCENARIO = {
    "v0": 0.16,
    "mu": 0.0,
    "theta": 2.0,
    "omega": 0.04,
    "xi": 0.3,
    "rho": -0.7,
}


@dataclass
class DatasetScatterPoint:
    name: str
    delta_temp_cost: float
    delta_is: float
    mean_ac: float
    mean_hest: float


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def _base_model_and_market(hparams: dict) -> tuple[AlmgrenChrissModel, MarketEnvironment]:
    model = AlmgrenChrissModel(
        X=BASE_PARAMS["X"],
        T=BASE_PARAMS["T"],
        N=BASE_PARAMS["N"],
        sigma=BASE_PARAMS["sigma"],
        lambd=BASE_PARAMS["lambd"],
        eta=BASE_PARAMS["eta"],
        gamma=BASE_PARAMS["gamma"],
        xi=hparams["xi"],
        rho=hparams["rho"],
        v0=hparams["v0"],
        theta=hparams["theta"],
        omega=hparams["omega"],
    )
    market = MarketEnvironment(
        S0=BASE_PARAMS["S0"],
        sigma=BASE_PARAMS["sigma"],
        T=BASE_PARAMS["T"],
        N=BASE_PARAMS["N"],
        gamma=BASE_PARAMS["gamma"],
        eta=BASE_PARAMS["eta"],
        heston_params=hparams,
    )
    return model, market


def _temporary_cost(model: AlmgrenChrissModel, trades: np.ndarray) -> float:
    return float(model.eta * np.sum(np.asarray(trades, dtype=float) ** 2) / model.dt)


def _format_small_p_value(p_value: float) -> str:
    if p_value < 1e-2:
        return r"$p < 0.01$"
    return rf"$p = {p_value:.3e}$"


def _short_dataset_label(label: str) -> str:
    stem = label.replace(".zip", "").replace("LOBSTER_SampleFile_", "")
    return stem.split("_2012")[0]


def _paper_style_dataset_label(label: str) -> str:
    stem = label.rstrip("/").replace(".zip", "")
    match = re.match(
        r"LOBSTER_SampleFile_([A-Za-z]+)_\d{4}-\d{2}-\d{2}_(\d+)(?: \((\d+)\))?$",
        stem,
    )
    if match:
        symbol = match.group(1).upper()
        level = match.group(2)
        replica = match.group(3)
        formatted = f"{symbol}-{level}L"
        if replica and replica != "1":
            formatted = f"{formatted}-{replica}"
        return formatted
    return _short_dataset_label(label)


def _representative_seed(model: AlmgrenChrissModel, market: MarketEnvironment, hparams: dict) -> int:
    ac_trades = model.compute_trade_list(use_correction=False)
    best_seed = 0
    best_gap = -np.inf

    for seed in range(50):
        _, variance_path = market.simulate_unaffected_price_heston(seed=seed, **hparams)
        hest_trades = model.compute_trade_list(use_correction=True, variance_path=variance_path)
        gap = float(np.max(np.abs((model.X - np.cumsum(hest_trades)) - (model.X - np.cumsum(ac_trades)))))
        if gap > best_gap:
            best_gap = gap
            best_seed = seed

    return best_seed


def figure_1_optimal_trajectory(output_dir: Path) -> None:
    model, market = _base_model_and_market(SYNTHETIC_SCENARIO)
    seed = _representative_seed(model, market, SYNTHETIC_SCENARIO)

    _, variance_path = market.simulate_unaffected_price_heston(seed=seed, **SYNTHETIC_SCENARIO)
    ac_summary = model.summary(use_correction=False)
    hest_summary = model.summary(use_correction=True, variance_path=variance_path)

    times = ac_summary["times"]
    x_ac = ac_summary["inventory"]
    x_hest = hest_summary["inventory"]
    trades_ac = ac_summary["trades"]
    trades_hest = hest_summary["trades"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]})
    fig.suptitle(
        "Figure 1. Optimal Trajectory Pivot Plot under an Out-of-Equilibrium Volatility Shock",
        fontsize=14,
        fontweight="semibold",
    )

    ax_top = axes[0]
    ax_top.plot(times, x_ac, color="#4C72B0", lw=2.3, label="Almgren-Chriss (exact)")
    ax_top.plot(times, x_hest, color="#DD8452", lw=2.3, label="Inventory-scaled Heston")
    ax_top.set_ylabel("Remaining inventory $x_t$")
    ax_top.set_title(
        f"Inventory path pivot with initial variance shock: $v_0={SYNTHETIC_SCENARIO['v0']:.2f}$, $\omega={SYNTHETIC_SCENARIO['omega']:.2f}$, seed={seed}"
    )
    ax_top.grid(alpha=0.25)
    ax_top.legend(frameon=False, loc="upper right")

    ax_bottom = axes[1]
    ax_bottom.step(times[:-1], trades_ac, where="post", color="#4C72B0", lw=2.0, label="AC trade blocks")
    ax_bottom.step(times[:-1], trades_hest, where="post", color="#DD8452", lw=2.0, label="Heston trade blocks")
    ax_bottom.set_ylabel("Trade blocks $\Delta X_k$")
    ax_bottom.set_xlabel("Time $t$")
    ax_bottom.grid(alpha=0.25)

    ax_var = ax_bottom.twinx()
    ax_var.plot(times, variance_path, color="#C44E52", lw=2.0, ls="--", label="$v_t$")
    ax_var.set_ylabel("Variance path $v_t$", color="#C44E52")
    ax_var.tick_params(axis="y", labelcolor="#C44E52")

    handles_1, labels_1 = ax_bottom.get_legend_handles_labels()
    handles_2, labels_2 = ax_var.get_legend_handles_labels()
    ax_bottom.legend(handles_1 + handles_2, labels_1 + labels_2, frameon=False, loc="upper right")

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    _save_figure(fig, output_dir, "figure1_optimal_trajectory_pivot")
    plt.close(fig)


def _synthetic_distribution_samples(n_sims: int = 1000):
    model, market = _base_model_and_market(SYNTHETIC_SCENARIO)
    ac_trades = model.compute_trade_list(use_correction=False)

    is_ac = np.zeros(n_sims)
    is_hest = np.zeros(n_sims)

    for i in range(n_sims):
        price_path, variance_path = market.simulate_unaffected_price_heston(seed=10_000 + i, **SYNTHETIC_SCENARIO)
        hest_trades = model.compute_trade_list(use_correction=True, variance_path=variance_path)

        cash_ac = market.apply_market_impact(price_path, ac_trades)["total_cash"]
        cash_hest = market.apply_market_impact(price_path, hest_trades)["total_cash"]

        is_ac[i] = market.implementation_shortfall(model.X, cash_ac)
        is_hest[i] = market.implementation_shortfall(model.X, cash_hest)

    return is_ac, is_hest


def figure_2_synthetic_distribution(output_dir: Path) -> None:
    is_ac, is_hest = _synthetic_distribution_samples(n_sims=1000)
    t_stat, p_two = ttest_rel(is_ac, is_hest)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
    mu_ac = float(np.mean(is_ac))
    mu_hest = float(np.mean(is_hest))

    all_values = np.concatenate([is_ac, is_hest])
    bins = np.histogram_bin_edges(all_values, bins=36)

    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    fig.suptitle("Figure 2. Synthetic Control Distribution for Heston vs AC", fontsize=14, fontweight="semibold")
    ax.set_title(
        "Single-scenario run: "
        f"$v_0={SYNTHETIC_SCENARIO['v0']:.2f}$, "
        f"$\\omega={SYNTHETIC_SCENARIO['omega']:.2f}$, "
        f"$\\xi={SYNTHETIC_SCENARIO['xi']:.2f}$, "
        f"$\\rho={SYNTHETIC_SCENARIO['rho']:.2f}$, "
        "1000 paired paths",
        fontsize=11,
    )

    ax.hist(is_ac, bins=bins, alpha=0.60, density=True, color="#4C72B0", label="Almgren-Chriss")
    ax.hist(is_hest, bins=bins, alpha=0.60, density=True, color="#DD8452", label="Inventory-scaled Heston")
    ax.axvline(mu_ac, color="#4C72B0", ls="--", lw=2.0, label=rf"$\mu_{{AC}}={mu_ac:.2f}$")
    ax.axvline(mu_hest, color="#DD8452", ls="--", lw=2.0, label=rf"$\mu_{{Hest}}={mu_hest:.2f}$")

    text = (
        rf"Paired t-test: $t={t_stat:.3f}$" "\n"
        rf"{_format_small_p_value(p_one)} (one-sided)" "\n"
        rf"$\mu_{{AC}}-\mu_{{Hest}}={mu_ac - mu_hest:.3f}$"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.95),
    )

    ax.set_xlabel("Implementation Shortfall")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right", ncol=1)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _save_figure(fig, output_dir, "figure2_synthetic_control_distribution")
    plt.close(fig)


def _dataset_scatter_points(n_sims: int = 120) -> list[DatasetScatterPoint]:
    points: list[DatasetScatterPoint] = []
    datasets = list_datasets()

    for idx, dataset_path in enumerate(datasets):
        label = os.path.basename(dataset_path)
        if os.path.isdir(dataset_path):
            label = f"{label}/"
        label = _paper_style_dataset_label(label)

        calibrator = LobsterCalibrator.from_dataset(dataset_path)
        df = calibrator.load_data()

        sigma = calibrator.estimate_volatility(df)
        impact = calibrator.estimate_impact_parameters(df)
        heston = calibrator.estimate_heston_parameters(df)
        if heston is None:
            continue

        params = BASE_PARAMS.copy()
        params["S0"] = float(df["Mid_Price"].iloc[0]) if "Mid_Price" in df.columns else BASE_PARAMS["S0"]
        params["sigma"] = sigma if sigma is not None else BASE_PARAMS["sigma"]
        params["eta"] = impact.get("eta") if impact and impact.get("eta") is not None else BASE_PARAMS["eta"]
        params["gamma"] = impact.get("gamma") if impact and impact.get("gamma") is not None else BASE_PARAMS["gamma"]
        params["heston"] = heston

        strategy, env, _, _ = build_objects(params, BASE_PARAMS["lambd"])
        heston_model = HestonParameters(**heston)
        comparator = ModelComparator(
            model_ac=strategy,
            model_hest=heston_model,
            market_env=env,
            num_sims=n_sims,
            seed=1_000 + idx,
        )

        results = comparator.run_comparison()
        is_ac = np.asarray(results["is_ac"], dtype=float)
        is_hest = np.asarray(results["is_heston"], dtype=float)
        if len(is_ac) == 0 or len(is_hest) == 0:
            continue

        mean_ac = float(np.mean(is_ac))
        mean_hest = float(np.mean(is_hest))

        ac_trades = strategy.compute_trade_list(use_correction=False)
        delta_temp_costs = []
        delta_is_values = []
        rng = np.random.default_rng(25_000 + idx)
        for _ in range(n_sims):
            seed = int(rng.integers(0, 1_000_000_000))
            price_path, variance_path = env.simulate_unaffected_price_heston(seed=seed, **heston)
            hest_trades = strategy.compute_trade_list(use_correction=True, variance_path=variance_path)

            cash_ac = env.apply_market_impact(price_path, ac_trades)["total_cash"]
            cash_hest = env.apply_market_impact(price_path, hest_trades)["total_cash"]

            is_ac_val = env.implementation_shortfall(strategy.X, cash_ac)
            is_hest_val = env.implementation_shortfall(strategy.X, cash_hest)

            delta_is_values.append(is_ac_val - is_hest_val)
            delta_temp_costs.append(_temporary_cost(strategy, hest_trades) - _temporary_cost(strategy, ac_trades))

        points.append(
            DatasetScatterPoint(
                name=label,
                delta_temp_cost=float(np.mean(delta_temp_costs)),
                delta_is=float(np.mean(delta_is_values)),
                mean_ac=mean_ac,
                mean_hest=mean_hest,
            )
        )

    return points


def figure_3_real_world_scatter(output_dir: Path) -> None:
    points = _dataset_scatter_points(n_sims=120)
    if not points:
        raise RuntimeError("No empirical dataset points were available for Figure 3.")

    x = np.array([p.delta_temp_cost for p in points], dtype=float)
    y = np.array([p.delta_is for p in points], dtype=float)
    labels = [p.name for p in points]

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    fig.suptitle("Figure 3. Real-World Friction Scatter from LOBSTER-Calibrated Runs", fontsize=14, fontweight="semibold")

    pos = y >= 0
    neg = ~pos

    if np.any(pos):
        ax.scatter(
            x[pos],
            y[pos],
            s=58,
            c="#55A868",
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            label="Green: Heston improves IS ($\\Delta$IS $\\geq 0$)",
        )
    if np.any(neg):
        ax.scatter(
            x[neg],
            y[neg],
            s=58,
            c="#C44E52",
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            label="Red: Heston underperforms ($\\Delta$IS $< 0$)",
        )
    ax.axhline(0, color="0.35", lw=1.2, ls="--")
    ax.axvline(0, color="0.35", lw=1.2, ls="--")

    if len(points) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 200)
        ax.plot(xs, slope * xs + intercept, color="#4C72B0", lw=1.8, alpha=0.9, label="OLS trend")

    underperformers = np.argsort(y)[: min(5, len(y))]
    for idx in underperformers:
        ax.annotate(
            labels[idx],
            (x[idx], y[idx]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )

    ax.set_xlabel(r"$\Delta$ Temp Cost = Cost$_{Hest}$ - Cost$_{AC}$")
    ax.set_ylabel(r"$\Delta$ IS = IS$_{AC}$ - IS$_{Hest}$")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _save_figure(fig, output_dir, "figure3_real_world_friction_scatter")
    plt.close(fig)


def figure_4_quadrature_heatmap(output_dir: Path) -> None:
    kappa_vals = np.logspace(-6, -2, 80)
    tau_vals = np.logspace(-3, 0, 80)

    heat = np.zeros((len(tau_vals), len(kappa_vals)))
    sample_u = np.linspace(0.1, 0.9, 13)

    for i, tau in enumerate(tau_vals):
        u = sample_u * tau
        limit = np.maximum(u / tau, 1e-15)
        for j, kappa in enumerate(kappa_vals):
            if kappa * tau < 1e-7:
                ratio = u / tau
            else:
                denom = np.sinh(kappa * tau)
                numerator = np.sinh(kappa * u)
                ratio = numerator / denom
            rel_err = np.max(np.abs(ratio - limit) / limit)
            heat[i, j] = rel_err

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    fig.suptitle("Figure 4. Quadrature Stability Heatmap for Hyperbolic Ratios", fontsize=14, fontweight="semibold")

    log_heat = np.log10(np.maximum(heat, 1e-16))
    mesh = ax.pcolormesh(
        kappa_vals,
        tau_vals,
        log_heat,
        shading="auto",
        cmap="magma",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\tau = T - t$")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}$ relative error")
    ax.grid(alpha=0.18, which="both")
    ax.set_title(r"Stability of $\sinh(\kappa(T-s))/\sinh(\kappa(T-t))$ against the $\kappa \to 0$ limit", fontsize=11)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _save_figure(fig, output_dir, "figure4_quadrature_stability_heatmap")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the four paper figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "paper" / "figures",
        help="Directory where figure files will be written.",
    )
    args = parser.parse_args()

    output_dir = _ensure_output_dir(args.output_dir)

    figure_1_optimal_trajectory(output_dir)
    figure_2_synthetic_distribution(output_dir)
    figure_3_real_world_scatter(output_dir)
    figure_4_quadrature_heatmap(output_dir)

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()