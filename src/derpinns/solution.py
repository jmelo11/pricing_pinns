from scipy.stats import norm
import numpy as np
import time
import torch
from derpinns.sampling import random_samples
from derpinns.collocations import OptionParameters


def compare_with_mc(model, params: OptionParameters, n_prices, n_simulations, dtype, device, seed=None):
    """
        Compares the NN output to MC simulations. Could be optimized but it works well enought for now.
    """
    if isinstance(params.rho, torch.Tensor):
        rho = params.rho.cpu().detach().numpy()
    else:
        rho = params.rho
    if isinstance(params.sigma, torch.Tensor):
        sigma = params.sigma.cpu().detach().numpy()
    else:
        sigma = params.sigma

    np.random.seed(seed)
    x0 = np.random.uniform(low=params.x_min, high=params.x_max,
                           size=(n_prices, params.n_assets))
    s0 = np.exp(x0)*params.strike

    with torch.no_grad():
        start = time.time()
        input = np.hstack([x0, np.full((n_prices, 1), params.tau)])
        input = torch.tensor(input, dtype=dtype, device=device)
        nn_prices = model(input)
        nn_prices = nn_prices.cpu().detach().numpy().flatten()*params.strike
        end = time.time()
        nn_time = end - start

    mc_prices = np.zeros(n_prices)

    L = np.linalg.cholesky(rho)

    start = time.time()
    for i in range(n_prices):
        Z = random_samples(n_simulations, params.n_assets, sampler='Halton')
        Z = norm.ppf(Z)
        correlated_Z = Z @ L.T
        drift = (params.r - 0.5 * sigma**2) * params.tau
        diffusion = sigma * np.sqrt(params.tau) * correlated_Z
        S_T = s0[i] * np.exp(drift + diffusion)
        payoffs = np.maximum(np.max(S_T, axis=1) - params.strike, 0)
        mc_prices[i] = np.exp(-params.r * params.tau) * np.mean(payoffs)
    end = time.time()
    mc_time = end - start

    results = {
        'nn_prices': nn_prices,
        'mc_prices': mc_prices,
        'abs_error': np.abs(mc_prices - nn_prices),
        'avg_error': np.mean(np.abs(mc_prices - nn_prices)),
        'max_error': np.max(np.abs(mc_prices - nn_prices)),
        'avg_mc_price': np.mean(mc_prices),
        'avg_nn_price': np.mean(nn_prices),
        'total_mc_time': mc_time,
        'total_nn_time': nn_time,
        'l2_rel_error':  np.sqrt(np.sum((mc_prices-nn_prices)**2) / np.sum(mc_prices**2)),
        'avg_mc_time': mc_time / n_prices,
        'avg_nn_time': nn_time / n_prices,
    }
    return results


def bs_call_price(S, K, r, sigma, T):
    """
        Black scholes price of a vanilla call.
    """
    if T <= 1e-14:
        return max(S - K, 0.0)  # at (very) near maturity, revert to intrinsic
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
