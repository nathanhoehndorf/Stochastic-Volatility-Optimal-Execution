"""
Run the full dataset sweep programmatically and print all statistical results
in a format suitable for pasting into the paper.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import ttest_rel
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from data.calibrator import LobsterCalibrator
from evaluation.comparator import ModelComparator
from core.AlmgrenChrissModel import AlmgrenChrissModel
from core.MarketEnvironment import MarketEnvironment
from main import (
    list_datasets, build_objects, HestonParameters,
    calculate_sharpe_like, get_data_dir
)

# ── sweep parameters ──────────────────────────────────────────────────────────
LAMBD    = 0.5
N_SIMS   = 500      # per dataset
SEED     = 42
BASE_PARAMS = {
    "X"     : 10_000,
    "N"     : 78,
    "T"     : 1.0,
    "sigma" : 0.1,
    "eta"   : 0.0000025,
    "gamma" : 0.0000005,
    "heston": {
        "v0": 0.04, "mu": 0.0, "theta": 2.0,
        "omega": 0.04, "xi": 0.3, "rho": -0.7,
    },
}

def run_one(dataset_path, base_params, lambd, n_sims, seed, idx):
    label = os.path.basename(dataset_path)
    try:
        cal = LobsterCalibrator.from_dataset(dataset_path)
        df  = cal.load_data()

        sigma  = cal.estimate_volatility(df)
        impact = cal.estimate_impact_parameters(df)
        heston = cal.estimate_heston_parameters(df)

        if heston is None:
            return {"ok": False, "dataset": label, "reason": "Heston calibration failed"}

        params = base_params.copy()
        params["S0"]    = float(df["Mid_Price"].iloc[0])
        params["sigma"] = sigma if sigma is not None else base_params["sigma"]
        params["eta"]   = impact.get("eta")   or base_params["eta"]
        params["gamma"] = impact.get("gamma") or base_params["gamma"]
        params["heston"] = heston

        strategy, env, _, _ = build_objects(params, lambd)
        hm = HestonParameters(**heston)

        comp = ModelComparator(
            model_ac=strategy, model_hest=hm,
            market_env=env, num_sims=n_sims, seed=seed + idx
        )
        results = comp.run_comparison()

        is_ac    = np.asarray(results["is_ac"],     dtype=float)
        is_hest  = np.asarray(results["is_heston"], dtype=float)
        is_ac    = is_ac[np.isfinite(is_ac)]
        is_hest  = is_hest[np.isfinite(is_hest)]

        if len(is_ac) < 2:
            return {"ok": False, "dataset": label, "reason": "Too few valid simulations"}

        mean_ac   = float(np.mean(is_ac))
        std_ac    = float(np.std(is_ac, ddof=1))
        mean_hest = float(np.mean(is_hest))
        std_hest  = float(np.std(is_hest, ddof=1))

        return {
            "ok": True, "dataset": label,
            "record": {
                "dataset":    label,
                "mean_ac":    mean_ac,
                "std_ac":     std_ac,
                "mean_hest":  mean_hest,
                "std_hest":   std_hest,
                "mean_diff":  mean_ac - mean_hest,
                "sharpe_ac":  calculate_sharpe_like(mean_ac, std_ac),
                "sharpe_hest":calculate_sharpe_like(mean_hest, std_hest),
                "rho":        heston.get("rho"),
            },
            "is_ac":    is_ac.tolist(),
            "is_hest":  is_hest.tolist(),
            "sv":       np.asarray(results.get("starting_vols", []), dtype=float).tolist(),
        }

    except Exception as e:
        return {"ok": False, "dataset": label,
                "reason": str(e), "traceback": traceback.format_exc()}


def main():
    datasets = list_datasets()
    print(f"Found {len(datasets)} datasets.\n")

    tasks = [(p, BASE_PARAMS, LAMBD, N_SIMS, SEED, i) for i, p in enumerate(datasets)]

    records, failures = [], []
    all_is_ac, all_is_hest, all_sv = [], [], []

    # run sequentially so output is visible and deterministic
    for task in tasks:
        label = os.path.basename(task[0])
        print(f"  → {label} ...", flush=True)
        r = run_one(*task)
        if r["ok"]:
            records.append(r["record"])
            all_is_ac.extend(r["is_ac"])
            all_is_hest.extend(r["is_hest"])
            all_sv.extend(r["sv"])
            print(f"    mean_ac={r['record']['mean_ac']:.2f}  mean_hest={r['record']['mean_hest']:.2f}")
        else:
            failures.append(label)
            print(f"    FAILED: {r['reason'][:80]}")

    print(f"\nDatasets evaluated: {len(records)}, failed: {len(failures)}")
    if failures:
        print("Failed:", ", ".join(failures))

    if not records:
        print("No valid records — aborting."); return

    # ── across-dataset paired t-test ─────────────────────────────────────────
    mean_ac_arr   = np.array([r["mean_ac"]    for r in records])
    mean_hest_arr = np.array([r["mean_hest"]  for r in records])
    sharpe_ac_arr = np.array([r["sharpe_ac"]  for r in records])
    sharpe_hest_arr = np.array([r["sharpe_hest"] for r in records])

    diff_valid = mean_ac_arr - mean_hest_arr
    t_stat, p_two = ttest_rel(mean_ac_arr, mean_hest_arr)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2

    # sharpe test
    sdiff = sharpe_hest_arr - sharpe_ac_arr
    ts, p2s = ttest_rel(sharpe_hest_arr, sharpe_ac_arr)
    p_sharpe = p2s / 2 if ts > 0 else 1.0 - p2s / 2

    # ── within-dataset combined arrays (all sims pooled across datasets) ──────
    is_ac_all   = np.array(all_is_ac)
    is_hest_all = np.array(all_is_hest)
    sv_all      = np.array(all_sv)

    from scipy.stats import wilcoxon, levene, ks_2samp

    tw, p_two_w = ttest_rel(is_ac_all, is_hest_all)
    p_ttest_w   = p_two_w / 2 if tw > 0 else 1.0 - p_two_w / 2

    diff_all = is_ac_all - is_hest_all
    nz = diff_all[diff_all != 0]
    if len(nz) > 0 and not np.allclose(diff_all, 0):
        w_stat, p_wilcox = wilcoxon(diff_all, alternative="greater")
    else:
        w_stat, p_wilcox = None, None

    lev_stat, p_lev2 = levene(is_ac_all, is_hest_all)
    p_lev = p_lev2 / 2 if np.var(is_ac_all) > np.var(is_hest_all) else 1.0 - p_lev2 / 2

    ks_stat, p_ks = ks_2samp(is_ac_all, is_hest_all, alternative="two-sided")

    # CVaR
    def cvar(a, alpha):
        v = np.quantile(a, alpha)
        return float(np.mean(a[a >= v]))

    cvar95_ac, cvar95_hest = cvar(is_ac_all, 0.95), cvar(is_hest_all, 0.95)
    cvar99_ac, cvar99_hest = cvar(is_ac_all, 0.99), cvar(is_hest_all, 0.99)

    # regime analysis
    sv_finite = np.isfinite(sv_all)
    sv_f = sv_all[sv_finite]
    ac_f = is_ac_all[sv_finite]
    hest_f= is_hest_all[sv_finite]
    if len(sv_f) > 30 and not np.allclose(sv_f, sv_f[0]):
        idx_sorted = np.argsort(sv_f)
        splits = np.array_split(idx_sorted, 3)
        regime_labels = ["Low", "Mid", "High"]
        regimes = {}
        for label, bucket in zip(regime_labels, splits):
            rd = ac_f[bucket] - hest_f[bucket]
            lo, hi = float(sv_f[bucket].min()), float(sv_f[bucket].max())
            nz_r = rd[rd != 0]
            if len(nz_r) > 1 and not np.allclose(rd, 0):
                w_r, p_r = wilcoxon(rd, alternative="greater")
            else:
                w_r, p_r = None, None
            regimes[label] = {
                "lo": lo, "hi": hi, "n": len(bucket),
                "median": float(np.median(rd)),
                "w": w_r, "p": p_r
            }
    else:
        regimes = None

    # ── pretty print ─────────────────────────────────────────────────────────
    sep = "─" * 68
    print(f"\n{'═'*68}")
    print("  FRESH SWEEP RESULTS  (ready to paste into unified_paper.tex)")
    print(f"{'═'*68}\n")

    print("PER-DATASET TABLE:")
    print(f"{'Dataset':<50} {'Mean_AC':>12} {'Mean_Hest':>12} {'Diff':>12} {'Sharpe_AC':>10} {'Sharpe_Hest':>12}")
    for r in sorted(records, key=lambda x: x["dataset"]):
        print(f"{r['dataset']:<50} {r['mean_ac']:12.2f} {r['mean_hest']:12.2f} {r['mean_diff']:12.2f} {r['sharpe_ac']:10.2f} {r['sharpe_hest']:12.2f}")

    print(f"\n{sep}\nACROSS-DATASET PAIRED t-TEST\n{sep}")
    print(f"Valid datasets  : {len(records)}")
    print(f"Mean diff (AC-Hest): {float(diff_valid.mean()):.5f}")
    print(f"t-statistic     : {t_stat:.4f}")
    print(f"p-value (1-sided): {p_one:.4f}")
    print(f"Reject H0       : {p_one < 0.05}")

    print(f"\n{sep}\nSHARPE-LIKE RELIABILITY\n{sep}")
    print(f"Mean Sharpe_AC   : {float(sharpe_ac_arr.mean()):.2f}")
    print(f"Mean Sharpe_Hest : {float(sharpe_hest_arr.mean()):.2f}")
    print(f"t-statistic      : {ts:.4f}")
    print(f"p-value (1-sided): {p_sharpe:.4f}")
    print(f"Reject H0        : {p_sharpe < 0.05}")

    print(f"\n{sep}\nWITHIN-DATASET (all sims pooled)  N={len(is_ac_all)}\n{sep}")
    print(f"Paired t-test  t={tw:.4f}  p={p_ttest_w:.4f}  reject={p_ttest_w < 0.05}")
    if w_stat is not None:
        print(f"Wilcoxon       W={w_stat:.1f}  p={p_wilcox:.4f}  reject={p_wilcox < 0.05}")
    else:
        print("Wilcoxon       all diffs zero — cannot compute")
    print(f"Levene         F={lev_stat:.4f}  p={p_lev:.4f}  reject={p_lev < 0.05}")
    print(f"  var_ac={float(np.var(is_ac_all, ddof=1)):.4f}  var_hest={float(np.var(is_hest_all, ddof=1)):.4f}")
    print(f"KS test        D={ks_stat:.4f}  p={p_ks:.4f}  reject={p_ks < 0.05}")
    print(f"CVaR 95%  AC={cvar95_ac:.2f}  Hest={cvar95_hest:.2f}  diff={cvar95_hest-cvar95_ac:.2f}")
    print(f"CVaR 99%  AC={cvar99_ac:.2f}  Hest={cvar99_hest:.2f}  diff={cvar99_hest-cvar99_ac:.2f}")

    if regimes:
        print(f"\n{sep}\nREGIME ANALYSIS\n{sep}")
        for lab, rv in regimes.items():
            p_str = f"{rv['p']:.4f}" if rv["p"] is not None else "N/A"
            print(f"{lab}-vol  vol=[{rv['lo']:.3f},{rv['hi']:.3f}]  n={rv['n']}  median={rv['median']:.2f}  p={p_str}")

    print(f"\n{'═'*68}\n")


if __name__ == "__main__":
    main()
