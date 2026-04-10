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

def generate_heston_path(S0, v0, params, T, steps, seed=None):
    """
    Generate one Heston price path and one variance path using Euler-Maruyama.

    Parameters
    ----------
    S0 : float
        Initial asset price
    v0 : float
        Initial variance
    params : dict
        Must contain:
            mu    : drift of price
            theta : mean reversion speed of variance
            omega : long-run variance level
            xi    : vol of vol
            rho   : correlation between price and variance shocks
    T : float
        Total time horizon
    steps : int
        Number of time steps
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    S : np.ndarray
        Simulated price path of length steps + 1
    v : np.ndarray
        Simulated variance path of length steps + 1
    """
    if seed is not None:
        np.random.seed(seed)

    mu = params["mu"]
    theta = params["theta"]
    omega = params["omega"]
    xi = params["xi"]
    rho = params["rho"]

    dt = T / steps
    sqrt_dt = np.sqrt(dt)

    # Arrays to store the path
    S = np.zeros(steps + 1)
    v = np.zeros(steps + 1)

    S[0] = S0  #setting the first element of the arrays to the initial variance
    v[0] = v0

    # Correlation matrix for [Z1, Z2]
    corr_matrix = np.array([
        [1.0, rho],
        [rho, 1.0]
    ])

    # Cholesky factor
    L = np.linalg.cholesky(corr_matrix) #numpy library really doing some heavy lifting here
    # returns either a lowerr or uppoer triangular matrix

    for t in range(steps):
        # Generate two independent standard normals
        z = np.random.randn(2) 

        # Correlate them
        z_corr = L @ z
        Z1 = z_corr[0]   # variance shock
        Z2 = z_corr[1]   # price shock

        # Keep variance nonnegative inside sqrt
        v_t = max(v[t], 0.0)

        # Variance update
        v[t + 1] = v[t] + theta * (omega - v_t) * dt + xi * np.sqrt(v_t) * sqrt_dt * Z1

        # Optional truncation so variance never goes negative
        v[t + 1] = max(v[t + 1], 0.0)

        # Price update
        S[t + 1] = S[t] + mu * S[t] * dt + np.sqrt(v_t) * S[t] * sqrt_dt * Z2

    return S, v
