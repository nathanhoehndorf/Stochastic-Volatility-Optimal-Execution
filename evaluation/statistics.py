import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, levene, ks_2samp, ttest_rel
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Compute CVaR (Expected Shortfall) at confidence level alpha.
    
    losses  : 1-D array of IS losses (lower = better hedger)
    alpha   : tail probability, e.g. 0.95 or 0.99
    """
    var = np.quantile(losses, alpha)
    return float(np.mean(losses[losses >= var]))

def _bootstrap_cvar_diff(ac: np.ndarray, hest: np.ndarray, alpha: float = 0.95, n_bootstrap: int = 10_000, seed: int = 42) -> dict:
    """Bootstrap test for H0: CVaR_hest >= CVaR_ac vs H1: CVaR_hest < CVaR_ac"""
    rng = np.random.default_rng(seed)
    n = len(ac)
    obs_diff = _cvar(hest, alpha) - _cvar(ac, alpha)

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0,n,size=n)
        boot_diffs[i] = _cvar(hest[idx], alpha) - _cvar(ac[idx], alpha)

    p_value = float(np.mean(boot_diffs <= obs_diff))
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "cvar_ac": _cvar(ac, alpha),
        "cvar_hest": _cvar(hest, alpha),
        "obs_diff": obs_diff,
        "bootstrap_ci": (ci_low, ci_high),
        "p_value": p_value,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main test suite
# ─────────────────────────────────────────────────────────────────────────────

def calculate_test_suite(is_ac: np.ndarray, is_hest: np.ndarray, starting_vols: Optional[np.ndarray] = None, rho_sweep: Optional[dict] = None, alpha_levels: tuple = (0.95, 0.99), n_regimes: int = 3, n_bootstrap: int = 10_000, alpha_test: float = 0.05) -> dict:
    """
    Run the full statistical comparison between AC and Heston IS hedging errors.
 
    Parameters
    ----------
    is_ac          : 1-D array of IS losses under the AC model.
    is_hest        : 1-D array of IS losses under the Heston model.
    starting_vols  : 1-D array of starting volatilities (same length), used for
                     regime analysis. Pass None to skip that section.
    rho_sweep      : dict mapping rho value → {'is_ac': ..., 'is_hest': ...}
                     for the leverage-effect sensitivity analysis. Pass None to skip.
    alpha_levels   : CVaR confidence levels to evaluate.
    n_regimes      : number of volatility buckets for regime analysis.
    n_bootstrap    : bootstrap replicates for CVaR comparison.
    alpha_test     : significance level for all hypothesis tests.
 
    Returns
    -------
    results : nested dict with all test outputs (and the figures produced).
    """

    is_ac = np.asarray(is_ac, dtype=float)
    is_hest = np.asarray(is_hest, dtype=float)
    diff = is_ac - is_hest

    results = {}

    # Mean test (paired t-test), H0: mu_{X-Y} = 0   H1: mu_{X-Y} > 0  (Heston has lower IS)
    t_stat, p_two = ttest_rel(is_ac, is_hest)
    p_ttest = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2

    results["paired_ttest"] = {
        "description": "Paired t-test: H0 mu(AC-Hest)=0 vs H1 > 0",
        "t_statistic": float(t_stat),
        "p_value_one_sided": float(p_ttest),
        "mean_diff": float(diff.mean()),
        "reject_H0": p_ttest < alpha_test
    }

    # Wilcoxon signed-rank test (non-parametric mean test)
    w_stat, p_wilcox = wilcoxon(diff, alternative="greater")
    results["wilcoxon_signed_rank"] = {
        "description": "Wilcoxon signed-rank: H0 median(AC-Hest)=0 vs H1 > 0",
        "statistic": float(w_stat),
        "p_value": float(p_wilcox),
        "reject_H0": p_wilcox < alpha_test
    }

    # Levene's test for equal variance, H0: sigma^2_AC = sigma^2_Hest   H1: sigma^2_AC > sigma^2_Hest
    lev_stat, p_lev_two = levene(is_ac, is_hest)
    p_levene = p_lev_two / 2 if np.var(is_ac) > np.var(is_hest) else 1.0 - p_lev_two / 2

    results["levene_variance"] = {
        "description":   "Levene test: H0 var(AC)=var(Hest) vs H1 var(AC)>var(Hest)",
        "statistic":     float(lev_stat),
        "p_value_one_sided": float(p_levene),
        "var_ac":        float(np.var(is_ac,   ddof=1)),
        "var_hest":      float(np.var(is_hest, ddof=1)),
        "reject_H0":     p_levene < alpha_test
    }

    # Kolmogorov-Smirnov test (overall distribution)
    ks_stat, p_ks = ks_2samp(is_ac, is_hest, alternative="two-sided")

    results["ks_test"] = {
        "description":   "Two-sample KS test: H0 F_AC = F_Hest",
        "statistic":     float(ks_stat),
        "p_value":       float(p_ks),
        "reject_H0":     p_ks < alpha_test
    }

    # CVaR / Expected Shortfall
    results["cvar"] = {}
    for alpha in alpha_levels:
        boot = _bootstrap_cvar_diff(is_ac, is_hest, alpha=alpha, n_bootstrap=n_bootstrap)
        results["cvar"][f"{int(alpha*100)}%"] = {
            "description":     f"CVaR at {int(alpha*100)}%: bootstrap test H0 CVaR_Hest>=CVaR_AC",
            **boot,
            "reject_H0":       boot["p_value"] < alpha_test,
        }

    # Regime analysis
    if starting_vols is not None:
        starting_vols = np.asarray(starting_vols, dtype=float)
        boundaries = np.percentile(starting_vols, np.linspace(0, 100, n_regimes + 1))
        regime_labels = ["low-vol", "mid-vol", "high-vol"] if n_regimes == 3 else \
                        [f"regime_{i}" for i in range(n_regimes)]
 
        results["regime_analysis"] = {}
        for i in range(n_regimes):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (starting_vols >= lo) & (starting_vols <= hi)
            ac_r, hest_r = is_ac[mask], is_hest[mask]
 
            if mask.sum() < 10:
                results["regime_analysis"][regime_labels[i]] = {"warning": "too few samples"}
                continue
 
            w_r, p_r = wilcoxon(ac_r - hest_r, alternative="greater")
            results["regime_analysis"][regime_labels[i]] = {
                "vol_range":   (float(lo), float(hi)),
                "n_samples":   int(mask.sum()),
                "median_diff": float(np.median(ac_r - hest_r)),
                "wilcoxon_stat": float(w_r),
                "p_value":     float(p_r),
                "reject_H0":   p_r < alpha_test,
            }

    # Leverage effect sensitivity
    if rho_sweep is not None:
        rhos      = sorted(rho_sweep.keys())
        mean_is   = {"ac": [], "hest": []}
        cvar95    = {"ac": [], "hest": []}
 
        for rho in rhos:
            d = rho_sweep[rho]
            ac_r, hest_r = np.asarray(d["is_ac"]), np.asarray(d["is_hest"])
            mean_is["ac"].append(float(ac_r.mean()))
            mean_is["hest"].append(float(hest_r.mean()))
            cvar95["ac"].append(_cvar(ac_r,   0.95))
            cvar95["hest"].append(_cvar(hest_r, 0.95))
 
        # Spearman correlation between rho and mean IS difference
        diffs  = np.array(mean_is["ac"]) - np.array(mean_is["hest"])
        sp_r, sp_p = stats.spearmanr(rhos, diffs)
 
        results["leverage_sensitivity"] = {
            "rhos":           rhos,
            "mean_is_ac":     mean_is["ac"],
            "mean_is_hest":   mean_is["hest"],
            "cvar95_ac":      cvar95["ac"],
            "cvar95_hest":    cvar95["hest"],
            "spearman_r":     float(sp_r),
            "spearman_p":     float(sp_p),
            "significant_rho_effect": sp_p < alpha_test,
        }
 
    # Figures
    figs = _make_figures(
        is_ac, is_hest, results,
        starting_vols=starting_vols,
        rho_sweep=rho_sweep,
    )
    results["figures"] = figs

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _make_figures(is_ac, is_hest, results, starting_vols=None, rho_sweep=None):
    figs = {}

    # Figure 1 – Distribution overview
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig1.suptitle("IS Distribution Comparison: AC vs Heston", fontsize=13)
 
    ax = axes[0]
    ax.hist(is_ac,   bins=60, alpha=0.6, label="AC",    density=True, color="#4C72B0")
    ax.hist(is_hest, bins=60, alpha=0.6, label="Heston",density=True, color="#DD8452")
    ax.set_xlabel("Importance-Sampled Loss"); ax.set_ylabel("Density")
    ax.set_title("IS Distributions"); ax.legend()
 
    ax = axes[1]
    diff = is_ac - is_hest
    ax.hist(diff, bins=60, color="#55A868", alpha=0.8, density=True)
    ax.axvline(0, color="k", lw=1.5, ls="--")
    ax.axvline(diff.mean(), color="red", lw=1.5, ls="-", label=f"Mean={diff.mean():.3g}")
    ax.set_xlabel("IS_AC − IS_Hest"); ax.set_ylabel("Density")
    ax.set_title("Paired Differences"); ax.legend()
 
    ax = axes[2]
    for alpha, color in [(0.95, "#4C72B0"), (0.99, "#DD8452")]:
        label_a = f"CVaR {int(alpha*100)}%"
        ax.bar(
            [f"AC\n{label_a}", f"Hest\n{label_a}"],
            [_cvar(is_ac, alpha), _cvar(is_hest, alpha)],
            color=color, alpha=0.7,
        )
    ax.set_ylabel("CVaR"); ax.set_title("CVaR Comparison")
 
    plt.tight_layout()
    figs["distribution_overview"] = fig1
 
    # Figure 2 – Regime analysis
    if starting_vols is not None and "regime_analysis" in results:
        regimes = results["regime_analysis"]
        valid   = {k: v for k, v in regimes.items() if "warning" not in v}
        if valid:
            fig2, ax = plt.subplots(figsize=(8, 4))
            names  = list(valid.keys())
            medians = [v["median_diff"] for v in valid.values()]
            colors  = ["#55A868" if m > 0 else "#C44E52" for m in medians]
            pvals   = [v["p_value"] for v in valid.values()]
            bars = ax.bar(names, medians, color=colors, alpha=0.8)
            for bar, pv in zip(bars, pvals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"p={pv:.3f}",
                    ha="center", va="bottom", fontsize=9,
                )
            ax.axhline(0, color="k", lw=1, ls="--")
            ax.set_title("Regime Analysis: Median IS Difference (AC − Hest) per Vol Bucket")
            ax.set_ylabel("Median(IS_AC − IS_Hest)  [+ = Heston better]")
            plt.tight_layout()
            figs["regime_analysis"] = fig2
 
    # Figure 3 – Leverage sensitivity
    if rho_sweep is not None and "leverage_sensitivity" in results:
        ls = results["leverage_sensitivity"]
        fig3, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig3.suptitle("Sensitivity to Leverage Effect (ρ)", fontsize=12)
 
        ax = axes[0]
        ax.plot(ls["rhos"], ls["mean_is_ac"],   "o-", label="AC",    color="#4C72B0")
        ax.plot(ls["rhos"], ls["mean_is_hest"], "s-", label="Heston",color="#DD8452")
        ax.set_xlabel("ρ"); ax.set_ylabel("Mean IS")
        ax.set_title("Mean IS vs ρ"); ax.legend()
 
        ax = axes[1]
        ax.plot(ls["rhos"], ls["cvar95_ac"],   "o-", label="AC CVaR 95%",    color="#4C72B0")
        ax.plot(ls["rhos"], ls["cvar95_hest"], "s-", label="Heston CVaR 95%",color="#DD8452")
        r_str = f"Spearman ρ={ls['spearman_r']:.3f}, p={ls['spearman_p']:.3f}"
        ax.set_title(f"CVaR 95% vs ρ\n{r_str}")
        ax.set_xlabel("ρ"); ax.set_ylabel("CVaR 95%"); ax.legend()
 
        plt.tight_layout()
        figs["leverage_sensitivity"] = fig3
 
    return figs

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helper
# ─────────────────────────────────────────────────────────────────────────────

def print_results(results: dict, alpha_test: float = 0.05) -> None:
    """Print a human-readable summary of calculate_test_suite output."""
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print("  STATISTICAL TEST SUITE: AC vs HESTON IS COMPARISON")
    print(f"{'═'*60}\n")
 
    # 1. Paired t-test
    r = results["paired_ttest"]
    print(f"{sep}\n1. Paired t-test (mean IS)\n{sep}")
    print(f"   Mean diff (AC−Hest) : {r['mean_diff']:.5f}")
    print(f"   t-statistic         : {r['t_statistic']:.4f}")
    print(f"   p-value (one-sided) : {r['p_value_one_sided']:.4f}")
    print(f"   Reject H0           : {r['reject_H0']}  (α={alpha_test})\n")
 
    # 2. Wilcoxon
    r = results["wilcoxon_signed_rank"]
    print(f"{sep}\n2. Wilcoxon Signed-Rank Test\n{sep}")
    print(f"   Statistic           : {r['statistic']:.4f}")
    print(f"   p-value (one-sided) : {r['p_value']:.4f}")
    print(f"   Reject H0           : {r['reject_H0']}\n")
 
    # 3. Levene
    r = results["levene_variance"]
    print(f"{sep}\n3. Levene Variance Test\n{sep}")
    print(f"   Var(AC)             : {r['var_ac']:.5f}")
    print(f"   Var(Hest)           : {r['var_hest']:.5f}")
    print(f"   Statistic           : {r['statistic']:.4f}")
    print(f"   p-value (one-sided) : {r['p_value_one_sided']:.4f}")
    print(f"   Reject H0           : {r['reject_H0']}\n")
 
    # 4. KS
    r = results["ks_test"]
    print(f"{sep}\n4. Two-Sample KS Test\n{sep}")
    print(f"   KS statistic        : {r['statistic']:.4f}")
    print(f"   p-value (two-sided) : {r['p_value']:.4f}")
    print(f"   Reject H0           : {r['reject_H0']}\n")
 
    # 5. CVaR
    print(f"{sep}\n5. CVaR / Expected Shortfall\n{sep}")
    for key, r in results["cvar"].items():
        print(f"   [{key}]  CVaR_AC={r['cvar_ac']:.5f}  CVaR_Hest={r['cvar_hest']:.5f}")
        print(f"          Diff={r['observed_diff']:.5f}  "
              f"95% CI=[{r['bootstrap_ci'][0]:.5f}, {r['bootstrap_ci'][1]:.5f}]")
        print(f"          p-value={r['p_value']:.4f}  Reject H0: {r['reject_H0']}\n")
 
    # 6. Regime analysis
    if "regime_analysis" in results:
        print(f"{sep}\n6. Regime Analysis (Volatility Buckets)\n{sep}")
        for regime, r in results["regime_analysis"].items():
            if "warning" in r:
                print(f"   {regime}: {r['warning']}")
                continue
            print(f"   {regime} (vol ∈ [{r['vol_range'][0]:.3f}, {r['vol_range'][1]:.3f}], "
                  f"n={r['n_samples']})")
            print(f"      Median diff={r['median_diff']:.5f}  "
                  f"p={r['p_value']:.4f}  Reject H0: {r['reject_H0']}\n")
 
    # 7. Leverage sensitivity
    if "leverage_sensitivity" in results:
        r = results["leverage_sensitivity"]
        print(f"{sep}\n7. Leverage-Effect Sensitivity (ρ sweep)\n{sep}")
        print(f"   Spearman r (ρ vs IS diff) : {r['spearman_r']:.4f}")
        print(f"   Spearman p-value          : {r['spearman_p']:.4f}")
        print(f"   Significant correlation   : {r['significant_rho_effect']}\n")
 
    print(f"{'═'*60}\n")

# Mean-testing for expected shortfall
# paired t-test, with null mu_{X-Y}=0 and alternative mu_{X-Y}>0, meaning heston has lower IS
# Wilcoxon signed-rank test

# Variance-testing for risk
# Levene's test for equal variance, with null sigma^2_X = sigma^2_Y and alternative sigma^2_X > sigma^2_Y, meaning heston has lower variance

# Testing overall distribution
# Two-sample Kolmogorov-Smirnov test, 

# Tail Risk Evaluation with Conditional Value at Risk (run CVaR at 95% and 99%, use bootstrapping to test difference in CVaR)
# Conditional State Testing with Regime Analysis (Bucket simulations based on starting volatility relative to long-term mean), run Wilcoxon Signed-rank test within each bucket to see if Heston outperforms AC in specific regimes

# Sensitivity to Leverage Effect: run parameter sweep from -0.8 to 0.0, plot mean IS an CVaR as a function of rho, if obvious graphical correlation isn't present then run tests



