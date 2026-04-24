import MonteCarloSimulator as m
import Backtester as b
import MarketEnvironment as me
import AlmgrenChrissModel as ac
import numpy as np


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
    num_values = get_int("Number of lambda values to test", 20)
    n_sims = get_int("Monte Carlo simulations per lambda", 1000)
    risk_penalty = get_float("Risk penalty on std implementation shortfall", 1.0)

    lambda_values = np.linspace(min_lambda, max_lambda, num_values)

    sim = m.MonteCarloSimulator(
        S0=params["S0"],
        X=params["X"],
        T=params["T"],
        N=params["N"],
        sigma=params["sigma"],
        eta=params["eta"],
        gamma=params["gamma"]
    )

    results_df = sim.run_lambda_grid(lambda_values, n_sims=n_sims, seed=42)

    results_df["objective"] = (
        results_df["mean_is"] + risk_penalty * results_df["std_is"]
    )

    best_row = results_df.loc[results_df["objective"].idxmin()]

    print("\nLambda grid results:")
    print(results_df)

    print("\nBest lambda:")
    print(best_row)

    return best_row, results_df


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


def main():
    params = get_base_parameters()

    while True:
        print("\n========== MAIN MENU ==========")
        print("1. Run single backtest")
        print("2. Run Monte Carlo for one lambda")
        print("3. Optimize lambda with grid search")
        print("4. Change base parameters")
        print("5. Quit")

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
            print("Goodbye.")
            break

        else:
            print("Invalid choice. Please choose 1-5.")


if __name__ == "__main__":
    main()