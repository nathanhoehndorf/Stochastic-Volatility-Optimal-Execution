import core.MonteCarloSimulator as m
import core.Backtester as b
import core.MarketEnvironment as me
import core.AlmgrenChrissModel as ac
from evaluation.comparator import ModelComparator
from evaluation.statistics import print_results
import numpy as np
import matplotlib.pyplot as plt

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


def get_base_parameters():
    print("\nEnter simulation parameters. Press Enter to use defaults.\n")

    params = {
        "S0": get_float("Initial stock price S0", 375),
        "X": get_float("Shares to execute X", 30),
        "T": get_float("Trading horizon T", 10),
        "N": get_int("Number of intervals N", 40),
        "sigma": get_float("Volatility sigma", 55),
        "eta": get_float("Temporary impact eta", 0.05),
        "gamma": get_float("Permanent impact gamma", 0.1),
    }

    return params


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
    heston_model = HestonParameters(
        v0=get_float("Initial variance (v0)", 0.04),
        mu=get_float("Drift (mu)", 0.0),
        theta=get_float("Mean reversion rate (theta/kappa)", 2.0),
        omega=get_float("Long-term variance (omega)", 0.04),
        xi=get_float("Volatility of volatility (xi)", 0.3),
        rho=get_float("Correlation (rho)", -0.7)
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

def main():
    params = get_base_parameters()

    while True:
        print("\n========== MAIN MENU ==========")
        print("1. Run single backtest")
        print("2. Run Monte Carlo for one lambda")
        print("3. Optimize lambda with grid search")
        print("4. Change base parameters")
        print("5. Run model comparison suite")
        print("6. Quit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            run_single_backtest(params)

        elif choice == "2":
            run_single_lambda_mc(params)

        elif choice == "3":
            optimize_lambda(params)

        elif choice == "4":
            params = get_base_parameters()

        elif choice == "5":
            run_model_comparison(params)

        elif choice == "6":
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Please choose 1-5.")


if __name__ == "__main__":
    main()