import os
import core.MonteCarloSimulator as m
import core.Backtester as b
import core.MarketEnvironment as me
import core.AlmgrenChrissModel as ac
from data.calibrator import LobsterCalibrator
from evaluation.comparator import ModelComparator
from evaluation.statistics import print_results
from scipy.stats import ttest_rel
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

def run_one_dataset_sweep_task(args):
    """
    Worker function for one dataset.

    This must be top-level so ProcessPoolExecutor can pickle it on Windows.
    """
    dataset_path, base_params, lambd, n_sims, seed, worker_index = args

    label = os.path.basename(dataset_path)
    if os.path.isdir(dataset_path):
        label += "/"

    try:
        calibrator = LobsterCalibrator.from_dataset(dataset_path)
        df = calibrator.load_data()

        sigma = calibrator.estimate_volatility(df)
        impact_params = calibrator.estimate_impact_parameters(df)
        heston_params = calibrator.estimate_heston_parameters(df)

        if heston_params is None:
            return {
                "ok": False,
                "dataset": label,
                "reason": "Heston parameter estimation failed.",
            }

        dataset_params = base_params.copy()
        dataset_params["S0"] = (
            float(df["Mid_Price"].iloc[0])
            if "Mid_Price" in df.columns
            else base_params["S0"]
        )
        dataset_params["sigma"] = sigma if sigma is not None else base_params["sigma"]
        dataset_params["eta"] = (
            impact_params.get("eta")
            if impact_params and impact_params.get("eta") is not None
            else base_params["eta"]
        )
        dataset_params["gamma"] = (
            impact_params.get("gamma")
            if impact_params and impact_params.get("gamma") is not None
            else base_params["gamma"]
        )
        dataset_params["heston"] = heston_params

        strategy, env, _, _ = build_objects(dataset_params, lambd)

        heston_model = HestonParameters(
            v0=heston_params["v0"],
            mu=heston_params["mu"],
            theta=heston_params["theta"],
            omega=heston_params["omega"],
            xi=heston_params["xi"],
            rho=heston_params["rho"],
        )

        comp = ModelComparator(
            model_ac=strategy,
            model_hest=heston_model,
            market_env=env,
            num_sims=n_sims,
            seed=seed + worker_index,
        )

        results = comp.run_comparison()

        is_ac = np.asarray(results["is_ac"], dtype=float)
        is_heston = np.asarray(results["is_heston"], dtype=float)

        is_ac = is_ac[np.isfinite(is_ac)]
        is_heston = is_heston[np.isfinite(is_heston)]

        if len(is_ac) < 2 or len(is_heston) < 2:
            return {
                "ok": False,
                "dataset": label,
                "reason": "Too few valid simulation outputs.",
            }

        mean_ac = float(np.mean(is_ac))
        std_ac = float(np.std(is_ac, ddof=1))
        mean_hest = float(np.mean(is_heston))
        std_hest = float(np.std(is_heston, ddof=1))

        record = {
            "dataset": label,
            "mean_ac": mean_ac,
            "std_ac": std_ac,
            "mean_hest": mean_hest,
            "std_hest": std_hest,
            "mean_diff": mean_ac - mean_hest,
            "sharpe_ac": calculate_sharpe_like(mean_ac, std_ac),
            "sharpe_hest": calculate_sharpe_like(mean_hest, std_hest),
            "rho": heston_params.get("rho"),
        }

        return {
            "ok": True,
            "dataset": label,
            "record": record,
        }

    except Exception as exc:
        return {
            "ok": False,
            "dataset": label,
            "reason": str(exc),
            "traceback": traceback.format_exc(),
        }

def get_data_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "data/"))

def unzip_datasets_once():
    data_dir = get_data_dir()
    marker_path = os.path.join(data_dir, ".unzipped_once")

    if not os.path.isdir(data_dir):
        print(f"No data directory found at: {data_dir}")
        return

    if os.path.exists(marker_path):
        return

    zip_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".zip")
    ]

    if not zip_files:
        print(f"No zip files found in: {data_dir}")
        with open(marker_path, "w") as marker:
            marker.write("No zip files were present on first run.\n")
        return

    print("\nUnzipping dataset archives for first-time setup...")

    for zip_name in zip_files:
        zip_path = os.path.join(data_dir, zip_name)
        extract_dir = os.path.join(data_dir, os.path.splitext(zip_name)[0])

        if os.path.isdir(extract_dir) and os.listdir(extract_dir):
            print(f"  Skipping {zip_name}: extracted folder already exists at {extract_dir}")
            continue

        os.makedirs(extract_dir, exist_ok=True)

        print(f"  Extracting {zip_name} -> {extract_dir}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    with open(marker_path, "w") as marker:
        marker.write("Zip files were extracted successfully.\n")

    print("Finished unzipping dataset archives.\n")



def plot_lambda_results(results):
    plt.figure()
    plt.plot(results["lambda"], results["mean_is"], marker="o", label="Mean IS")
    plt.plot(results["lambda"], results["std_is"], marker="o", label="Std IS")
    plt.plot(results["lambda"], results["objective"], marker="o", label="Objective")

    plt.xlabel("Lambda")
    plt.ylabel("Value")
    plt.title("Lambda Optimization Results")
    plt.legend()
    plt.grid(True)
    plt.show()

def display_lambda_results(results, best):
    print("\n========== LAMBDA OPTIMIZATION RESULTS ==========")

    print("\nBest lambda:")
    print(f"  lambda: {best['lambda']:.6f}")
    print(f"  objective: {best['objective']:.6f}")
    print(f"  mean implementation shortfall: {best['mean_is']:.6f}")
    print(f"  std implementation shortfall: {best['std_is']:.6f}")
    print(f"  kappa: {best['kappa']:.6f}")

    print("\nFull lambda grid:")
    display_cols = ["lambda", "mean_is", "std_is", "objective", "kappa"]
    print(results[display_cols].to_string(index=False))

def get_float(prompt, default):
    user_input = input(f"{prompt} [{default}]: ").strip()
    if user_input == "":
        return default
    return float(user_input)


def get_int(prompt, default):
    user_input = input(f"{prompt} [{default}]: ").strip()
    if user_input == "":
        return default  
    return int(user_input)


def list_datasets():
    data_dir = get_data_dir()

    if not os.path.isdir(data_dir):
        return []

    entries = sorted(os.listdir(data_dir))
    datasets = []
    seen_bases = set()

    # Prefer extracted directories over zip archives with the same base name
    for name in entries:
        path = os.path.join(data_dir, name)
        base_name, ext = os.path.splitext(name)

        if os.path.isdir(path):
            datasets.append(os.path.abspath(path))
            seen_bases.add(name)
            seen_bases.add(base_name)

    for name in entries:
        path = os.path.join(data_dir, name)
        base_name, ext = os.path.splitext(name)

        if ext.lower() == '.zip' and base_name not in seen_bases:
            datasets.append(os.path.abspath(path))

    return datasets


def choose_dataset():
    datasets = list_datasets()

    if not datasets:
        print("No LOBSTER dataset archives or directories found in data.")
        return None

    print("\nChoose a LOBSTER dataset for parameter calibration:")

    for idx, path in enumerate(datasets, start=1):
        label = os.path.basename(path)
        if os.path.isdir(path):
            label += "/"
        print(f"{idx}. {label}")

    print("0. Skip dataset calibration")

    choice = get_int("Select dataset", 1)

    if choice <= 0 or choice > len(datasets):
        print("Skipping dataset calibration.")
        return None

    return datasets[choice - 1]


def estimate_parameters_from_dataset(dataset_path):
    print(f"\nEstimating parameters from dataset: {os.path.basename(dataset_path)}")
    calibrator = LobsterCalibrator.from_dataset(dataset_path)
    df = calibrator.load_data()

    sigma = calibrator.estimate_volatility(df)
    impact_params = calibrator.estimate_impact_parameters(df)
    heston_params = calibrator.estimate_heston_parameters(df)

    if heston_params is not None:
        heston_params['xi'] = min(heston_params['xi'], 0.8)
        heston_params['omega'] = max(heston_params['omega'], 0.1)
        heston_params['v0'] = max(heston_params['v0'], 0.0001)
        heston_params['theta'] = max(heston_params['theta'], 0.0001)

    estimated_eta = impact_params.get("eta") if impact_params else None
    estimated_gamma = impact_params.get("gamma") if impact_params else None

    defaults = {
        "S0": float(df["Mid_Price"].iloc[0]) if "Mid_Price" in df.columns else 100,
        "sigma": sigma if sigma is not None else 0.06,
        "eta": estimated_eta if estimated_eta is not None else 0.005,
        "gamma": estimated_gamma if estimated_gamma is not None else 0.0000047,
        "heston": heston_params if heston_params is not None else {
            "v0": 0.04,
            "mu": 0.0,
            "theta": 2.0,
            "omega": 0.04,
            "xi": 0.3,
            "rho": -0.7,
        }
    }

    print("\n========== ESTIMATED CALIBRATION DEFAULTS ==========")

    print(f"  S0     = {defaults['S0']:.4f}")
    print(f"  sigma  = {defaults['sigma']:.8f}")

    if defaults["eta"] is not None:
        print(f"  eta    = {defaults['eta']:.12e}")
    else:
        print("  eta    = estimation failed, using fallback default")

    if defaults["gamma"] is not None:
        print(f"  gamma  = {defaults['gamma']:.12e}")
    else:
        print("  gamma  = estimation failed, using fallback default")

    if "heston" in defaults and defaults["heston"] is not None:
        print("\n  Heston parameters:")
        for key, value in defaults["heston"].items():
            print(f"    {key:<6} = {value:.12e}")

    print("===================================================\n")

    return defaults


def get_base_parameters(defaults=None):
    defaults = defaults or {}
    print("\nEnter simulation parameters. Press Enter to use defaults.\n")

    params = {
        "S0": get_float("Initial stock price S0", defaults.get("S0", 100)),
        "X": get_float("Shares to execute X", defaults.get("X", 1_000_000)),
        "N": get_int("Number of intervals N", defaults.get("N", 78)),
        "T": get_float("Trading horizon T", defaults.get("T", 1)),
        "sigma": get_float("Volatility sigma", defaults.get("sigma", 0.1)),
        "eta": get_float("Temporary impact eta", defaults.get("eta", 0.0000025)),
        "gamma": get_float("Permanent impact gamma", defaults.get("gamma", 0.0000005)),
        "heston": defaults.get("heston", {
            "v0": 0.04,
            "mu": 0.0,
            "theta": 2.0,
            "omega": 0.04,
            "xi": 0.3,
            "rho": -0.7,
        })
    }

    return params


def calculate_sharpe_like(mean_is: float, std_is: float) -> float:
    """Compute a Sharpe-like reliability ratio from mean IS and its standard deviation."""
    if std_is <= 0:
        return float('inf') if mean_is < 0 else float('-inf') if mean_is > 0 else 0.0
    return float(-mean_is / std_is)


def build_objects(params, lambd):
    strategy = ac.AlmgrenChrissModel(
        X=params["X"],
        T=params["T"],
        N=params["N"],
        sigma=params["sigma"],
        lambd=lambd,
        eta=params["eta"],
        gamma=params["gamma"]
    )

    env = me.MarketEnvironment(
        S0=params["S0"],
        sigma=params["sigma"],
        T=params["T"],
        N=params["N"],
        gamma=params["gamma"],
        eta=params["eta"]
    )

    sim = m.MonteCarloSimulator(
        S0=params["S0"],
        X=params["X"],
        T=params["T"],
        N=params["N"],
        sigma=params["sigma"],
        eta=params["eta"],
        gamma=params["gamma"]
    )

    back = b.Backtester(strategy_model=strategy, market_env=env)

    return strategy, env, sim, back

def optimize_lambda(params):
    print("\nLambda optimization")

    min_lambda = get_float("Minimum lambda", 0.01)
    max_lambda = get_float("Maximum lambda", 2.0)
    num_values = get_int("Number of lambda values per search", 20)
    n_sims = get_int("Monte Carlo simulations per lambda", 1000)
    risk_penalty = get_float("Risk penalty on std implementation shortfall", 1.0)

    sim = m.MonteCarloSimulator(
        S0=params["S0"],
        X=params["X"],
        T=params["T"],
        N=params["N"],
        sigma=params["sigma"],
        eta=params["eta"],
        gamma=params["gamma"]
    )

    # ----- Step 1: coarse search -----
    coarse_lambda_values = np.linspace(min_lambda, max_lambda, num_values)

    coarse_results = sim.run_lambda_grid(
        coarse_lambda_values,
        n_sims=n_sims,
        seed=42
    )

    coarse_results["objective"] = (
        coarse_results["mean_is"] + risk_penalty * coarse_results["std_is"]
    )

    coarse_best = coarse_results.loc[coarse_results["objective"].idxmin()]
    best_lambda = coarse_best["lambda"]

    # ----- Step 2: refined search around best coarse lambda -----
    refine_width = (max_lambda - min_lambda) / num_values

    refined_min = max(min_lambda, best_lambda - refine_width)
    refined_max = min(max_lambda, best_lambda + refine_width)

    refined_lambda_values = np.linspace(refined_min, refined_max, num_values)

    refined_results = sim.run_lambda_grid(
        refined_lambda_values,
        n_sims=n_sims,
        seed=123
    )

    refined_results["objective"] = (
        refined_results["mean_is"] + risk_penalty * refined_results["std_is"]
    )

    best_row = refined_results.loc[refined_results["objective"].idxmin()]

    print("\nCoarse search best lambda:")
    print(coarse_best)

    print("\nRefined search range:")
    print(f"{refined_min:.6f} to {refined_max:.6f}")

    display_lambda_results(refined_results, best_row)

    show_plot = input("\nShow plot? (y/n): ").strip().lower()

    if show_plot == "y":
        plot_lambda_results(refined_results)

    return best_row, refined_results

def run_single_backtest(params):
    lambd = get_float("Lambda", 0.5)

    strategy, env, sim, back = build_objects(params, lambd)

    log_df, summary = back.run(seed=42)

    print("\nBacktest log:")
    print(log_df)

    print("\nBacktest summary:")
    print(summary)


def run_single_lambda_mc(params):
    lambd = get_float("Lambda", 0.5)
    n_sims = get_int("Number of simulations", 1000)

    _, _, sim, _ = build_objects(params, lambd)

    result = sim.run_single_lambda(lambd, n_sims=n_sims, seed=42)

    print("\nMonte Carlo result:")
    print(f"lambda: {result['lambda']}")
    print(f"kappa: {result['kappa']}")
    print(f"mean implementation shortfall: {result['mean_is']}")
    print(f"std implementation shortfall: {result['std_is']}")
    print(f"variance implementation shortfall: {result['var_is']}")

class HestonParameters:
    def __init__(self, v0, mu, theta, omega, xi, rho):
        self.v0 = v0
        self.mu = mu
        self.theta = theta
        self.omega = omega
        self.xi = xi
        self.rho = rho

def run_model_comparison(params):
    print("\n========== HESTON VS AC MODEL COMPARISON ==========")
    lambd = get_float("Lambda for AC strategy", 0.5)
    n_sims = get_int("Number of simulations for comparison", 1000)

    print("\nEnter Heston Model Parameters:")
    heston_defaults = params.get("heston", {})
    heston_model = HestonParameters(
        v0=get_float("Initial variance (v0)", heston_defaults.get("v0", 0.04)),
        mu=get_float("Drift (mu)", heston_defaults.get("mu", 0.0)),
        theta=get_float("Mean reversion rate (theta/kappa)", heston_defaults.get("theta", 2.0)),
        omega=get_float("Long-term variance (omega)", heston_defaults.get("omega", 0.04)),
        xi=get_float("Volatility of volatility (xi)", heston_defaults.get("xi", 0.3)),
        rho=get_float("Correlation (rho)", heston_defaults.get("rho", -0.7))
    )

    # Build base objects
    strategy, env, _, _ = build_objects(params, lambd)

    # Initialize the Comparator
    comp = ModelComparator(
        model_ac=strategy,
        model_hest=heston_model,
        market_env=env,
        num_sims=n_sims,
        seed=42
    )

    print(f"\nRunning {n_sims} paired simulations. This may take a moment...")
    
    # Run the suite and print results
    results = comp.run_comparison()
    print_results(results)

    # Show generated figures
    if "figures" in results:
        show_plot = input("\nShow evaluation plots? (y/n): ").strip().lower()
        if show_plot == "y":
            for name, fig in results["figures"].items():
                fig.show()
            plt.show()


def run_dataset_sweep(params):
    datasets = list_datasets()
    if not datasets:
        print("No dataset archives or directories found to sweep.")
        return

    print("\n========== LOBSTER DATASET SWEEP ==========")
    print(
        "This sweep will calibrate each available dataset, run paired AC vs Heston "
        "simulations, and compare results across all samples."
    )

    lambd = get_float("Lambda for AC strategy", 0.5)
    n_sims = get_int("Number of simulations per dataset", 500)
    seed = int(get_float("Random seed", 42))

    default_workers = max(1, min(len(datasets), max(1, (os.cpu_count() or 2) - 1)))
    max_workers = get_int("Parallel workers", default_workers)

    max_workers = max(1, min(max_workers, len(datasets)))

    print(f"\nRunning dataset sweep with {max_workers} parallel worker(s).")
    print(f"Datasets found: {len(datasets)}")

    sweep_records = []
    skipped = []

    tasks = [
        (dataset_path, params, lambd, n_sims, seed, idx)
        for idx, dataset_path in enumerate(datasets)
    ]

    if max_workers == 1:
        # Sequential fallback. Useful for debugging.
        for task in tasks:
            dataset_path = task[0]
            label = os.path.basename(dataset_path)
            if os.path.isdir(dataset_path):
                label += "/"

            print(f"\n--- Dataset: {label} ---")

            result = run_one_dataset_sweep_task(task)

            if result["ok"]:
                print(f"  Finished dataset: {result['dataset']}")
                sweep_records.append(result["record"])
            else:
                print(f"  Skipping dataset {result['dataset']}: {result['reason']}")
                skipped.append(result["dataset"])

    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_label = {}

            for task in tasks:
                dataset_path = task[0]
                label = os.path.basename(dataset_path)
                if os.path.isdir(dataset_path):
                    label += "/"

                future = executor.submit(run_one_dataset_sweep_task, task)
                future_to_label[future] = label

            completed = 0

            for future in as_completed(future_to_label):
                label = future_to_label[future]
                completed += 1

                try:
                    result = future.result()
                except Exception as exc:
                    print(f"\n[{completed}/{len(tasks)}] {label}: failed with executor error: {exc}")
                    skipped.append(label)
                    continue

                if result["ok"]:
                    print(f"\n[{completed}/{len(tasks)}] {result['dataset']}: finished")
                    sweep_records.append(result["record"])
                else:
                    print(f"\n[{completed}/{len(tasks)}] {result['dataset']}: skipped - {result['reason']}")
                    skipped.append(result["dataset"])

    if not sweep_records:
        print("No datasets produced valid sweep results.")
        return

    mean_ac_arr = np.array([r["mean_ac"] for r in sweep_records], dtype=float)
    mean_hest_arr = np.array([r["mean_hest"] for r in sweep_records], dtype=float)
    sharpe_ac_arr = np.array([r["sharpe_ac"] for r in sweep_records], dtype=float)
    sharpe_hest_arr = np.array([r["sharpe_hest"] for r in sweep_records], dtype=float)
    diff_arr = mean_ac_arr - mean_hest_arr

    valid_mean_mask = (
        np.isfinite(mean_ac_arr)
        & np.isfinite(mean_hest_arr)
    )

    mean_ac_valid = mean_ac_arr[valid_mean_mask]
    mean_hest_valid = mean_hest_arr[valid_mean_mask]
    diff_valid = mean_ac_valid - mean_hest_valid

    if len(diff_valid) < 2:
        t_stat = np.nan
        p_ttest = np.nan
        ttest_warning = "Across-dataset paired t-test skipped: need at least 2 valid datasets."
    elif np.allclose(diff_valid, diff_valid[0]):
        t_stat = np.nan
        p_ttest = np.nan
        ttest_warning = "Across-dataset paired t-test skipped: paired differences have zero variance."
    else:
        t_stat, p_two = ttest_rel(mean_ac_valid, mean_hest_valid)
        p_ttest = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2
        ttest_warning = None

    valid_sharpe_mask = (
        np.isfinite(sharpe_ac_arr)
        & np.isfinite(sharpe_hest_arr)
    )

    sharpe_ac_valid = sharpe_ac_arr[valid_sharpe_mask]
    sharpe_hest_valid = sharpe_hest_arr[valid_sharpe_mask]
    sharpe_diff_valid = sharpe_hest_valid - sharpe_ac_valid

    if len(sharpe_diff_valid) < 2:
        t_s = np.nan
        p_ttest_sharpe = np.nan
        sharpe_warning = "Sharpe-like paired t-test skipped: need at least 2 valid datasets."
    elif np.allclose(sharpe_diff_valid, sharpe_diff_valid[0]):
        t_s = np.nan
        p_ttest_sharpe = np.nan
        sharpe_warning = "Sharpe-like paired t-test skipped: paired differences have zero variance."
    else:
        # Alternative: Heston reliability > AC reliability.
        t_s, p_two_s = ttest_rel(sharpe_hest_valid, sharpe_ac_valid)
        p_ttest_sharpe = p_two_s / 2 if t_s > 0 else 1.0 - p_two_s / 2
        sharpe_warning = None

    print("\n========== SWEEP SUMMARY AC VS HESTON ==========")
    print(f"Datasets evaluated  : {len(sweep_records)}")
    if skipped:
        print(f"Datasets skipped    : {len(skipped)} -> {', '.join(skipped)}")

    print("\nPer-dataset mean implementation shortfall and reliability:")
    print(
        f"{'Dataset':<35} "
        f"{'Mean_AC':>12} "
        f"{'Mean_Hest':>12} "
        f"{'Diff':>12} "
        f"{'Sharpe_AC':>12} "
        f"{'Sharpe_Hest':>12}"
    )

    for record in sorted(sweep_records, key=lambda r: r["dataset"]):
        print(
            f"{record['dataset']:<35} "
            f"{record['mean_ac']:12.5f} "
            f"{record['mean_hest']:12.5f} "
            f"{record['mean_diff']:12.5f} "
            f"{record['sharpe_ac']:12.5f} "
            f"{record['sharpe_hest']:12.5f}"
        )

    print("\n--- Across-dataset paired t-test ---")
    print(f"Valid datasets for mean test: {len(diff_valid)}")
    print(f"Mean difference (AC−Hest) across datasets: {diff_valid.mean():.5f}")

    if ttest_warning is not None:
        print(f"Warning: {ttest_warning}")
    else:
        print(f"t-statistic: {t_stat:.4f}")
        print(f"one-sided p-value: {p_ttest:.4f}")
        print(f"Heston has lower mean IS across datasets: {p_ttest < 0.05}")

    print("\n--- Sharpe-like reliability comparison ---")
    print(f"Valid datasets for Sharpe test: {len(sharpe_diff_valid)}")
    print(f"Mean Sharpe_AC: {sharpe_ac_valid.mean():.5f}")
    print(f"Mean Sharpe_Hest: {sharpe_hest_valid.mean():.5f}")

    if sharpe_warning is not None:
        print(f"Warning: {sharpe_warning}")
    else:
        print(f"t-statistic (Sharpe): {t_s:.4f}")
        print(f"one-sided p-value (Sharpe): {p_ttest_sharpe:.4f}")
        print(f"Heston has higher reliability across datasets: {p_ttest_sharpe < 0.05}")

    print("\nSweep complete. Use the model comparison suite for detailed analysis on a single parameter set.")

def main():
    unzip_datasets_once()

    selected_dataset = choose_dataset()
    defaults = None
    if selected_dataset is not None:
        defaults = estimate_parameters_from_dataset(selected_dataset)

    params = get_base_parameters(defaults=defaults)

    while True:
        print("\n========== MAIN MENU ==========")
        print("1. Run single backtest")
        print("2. Run Monte Carlo for one lambda")
        print("3. Optimize lambda with grid search (find most optimal lambda between specified range)")
        print("4. Change base parameters")
        print("5. Run model comparison suite")
        print("6. Run dataset sweep across all LOBSTER samples")
        print("7. Quit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            run_single_backtest(params)

        elif choice == "2":
            run_single_lambda_mc(params)

        elif choice == "3":
            optimize_lambda(params)

        elif choice == "4":
            params = get_base_parameters(defaults=params)

        elif choice == "5":
            run_model_comparison(params)

        elif choice == "6":
            run_dataset_sweep(params)

        elif choice == "7":
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Please choose 1-7.")


if __name__ == "__main__":
    main()