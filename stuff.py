import numpy as np
import pandas as pd
import AlmgrenChrissModel
import MarketEnvironment
from concurrent.futures import ProcessPoolExecutor
def mc_worker(args): #used for the parallel batching of the montecarlo simulations
    S0, X, T, N, sigma, eta, gamma, lambd, n_sims, seed = args

    rng = np.random.default_rng(seed)
    dt = T / N
    sqrt_dt = np.sqrt(dt)

    times = np.linspace(0, T, N + 1)
    kappa = np.sqrt((lambd * sigma**2) / eta)

    if np.isclose(kappa, 0):
        inventory = X * (1 - times / T)
    else:
        inventory = X * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)

    trades = inventory[:-1] - inventory[1:]
    cumulative_sold_before = np.concatenate([[0], np.cumsum(trades[:-1])])

    Z = rng.standard_normal((n_sims, N)) # is this the actual distribution that we want to be using?
    dP = sigma * sqrt_dt * Z

    P = np.zeros((n_sims, N + 1))
    P[:, 0] = S0
    P[:, 1:] = S0 + np.cumsum(dP, axis=1)

    P_start = P[:, :-1]
    perm_prices = P_start - gamma * cumulative_sold_before
    exec_prices = perm_prices - eta * (trades / dt)

    cashflows = exec_prices * trades
    total_cash = np.sum(cashflows, axis=1)

    return X * S0 - total_cash

# this is the parallel batching of the montecarlo simulations to hopefully speed up the process a decent bit
def monte_carlo_is_parallel(S0, X, T, N, sigma, eta, gamma, lambd,
                            n_sims=1000, n_workers=4, seed=42): # likely don't want seed hard coded
    sims_per_worker = [n_sims // n_workers] * n_workers
    for i in range(n_sims % n_workers):
        sims_per_worker[i] += 1

    args_list = []
    for i, chunk in enumerate(sims_per_worker):
        args_list.append((S0, X, T, N, sigma, eta, gamma, lambd, chunk, seed + i))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(mc_worker, args_list))

    return np.concatenate(results)