from scipy.stats import norm, multivariate_normal
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
import time
import torch
from derpinns.sampling import random_samples


@torch.no_grad()
def compare_with_mc(
    model,
    params,
    n_prices: int,
    n_simulations: int,
    dtype=torch.float32,
    device=torch.device("cpu"),
    seed=None,
):
    """
        Vectorised NN-vs-MC comparison.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    r = torch.as_tensor(params.r,   dtype=dtype, device=device)
    tau = torch.as_tensor(params.tau, dtype=dtype, device=device)

    rho = torch.as_tensor(params.rho,   dtype=dtype, device=device)
    sigma = torch.as_tensor(params.sigma, dtype=dtype, device=device)

    x0 = torch.empty((n_prices, params.n_assets),
                     dtype=dtype, device=device).uniform_(params.x_min,
                                                          params.x_max)
    s0 = torch.exp(x0) * params.strike

    # not implmeneted in mps, use numpy cholesky instead
    if device.type == "mps":
        rho = rho.cpu().numpy()
        L = np.linalg.cholesky(rho)
        L = torch.as_tensor(L, dtype=dtype, device=device)
    else:
        L = torch.linalg.cholesky(rho)

    drift = (params.r - 0.5 * sigma**2) * params.tau
    vol_sqrt = sigma * np.sqrt(params.tau)

    t0 = time.perf_counter()
    inp = torch.cat([x0, torch.full((n_prices, 1), params.tau,
                                    dtype=dtype, device=device)], dim=1)
    nn_prices = model(inp).squeeze() * params.strike   # (n_prices,)
    nn_time = time.perf_counter() - t0

    t0 = time.perf_counter()

    Z = random_samples(n_prices * n_simulations, params.n_assets,
                       sampler="Sobol")
    Z = torch.as_tensor(norm.ppf(Z), dtype=dtype, device=device)

    # apply correlation
    Z = Z @ L.T

    Z = Z.view(n_prices, n_simulations, params.n_assets)

    diffusion = vol_sqrt * Z          # (n_prices, n_sim, n_assets)
    ST = s0.unsqueeze(1) * torch.exp(drift + diffusion)

    payoffs = torch.clamp(ST.max(dim=-1).values - params.strike, min=0)
    mc_prices = torch.exp(-r * tau) * payoffs.mean(dim=1)
    mc_time = time.perf_counter() - t0

    nn_prices = nn_prices.cpu().numpy()
    mc_prices = mc_prices.cpu().numpy()

    abs_err = np.abs(mc_prices - nn_prices)
    results = dict(
        nn_prices=nn_prices,
        mc_prices=mc_prices,
        abs_error=abs_err,
        avg_error=abs_err.mean(),
        max_error=abs_err.max(),
        avg_mc_price=mc_prices.mean(),
        avg_nn_price=nn_prices.mean(),
        total_mc_time=mc_time,
        total_nn_time=nn_time,
        l2_rel_error=np.linalg.norm(mc_prices - nn_prices) /
        np.linalg.norm(mc_prices),
        avg_mc_time=mc_time / n_prices,
        avg_nn_time=nn_time / n_prices,
    )
    return results


def bivariate_normal_cdf(x, y, rho):
    """
    Bivariate normal CDF. Uses scipy's multivariate_normal.
    """
    cov = [[1.0, rho], [rho, 1.0]]
    return multivariate_normal(mean=[0, 0], cov=cov).cdf([x, y])


def call_on_minimum(
    V, H, F,         # asset spots V,H, strike F
    r,               # risk-free rate
    sigmaV, sigmaH,  # volatilities
    rhoVH,           # correlation between V,H
    T,               # final maturity
    t                # current time in [0,T]
):
    """
    Call on minimum of two assets V,H, with strike F.
    """
    tau = T - t
    eps = 1e-14
    if tau < eps:
        return max(min(V, H) - F, 0.0)

    sigma_sq = sigmaV**2 + sigmaH**2 - 2*rhoVH*sigmaV*sigmaH
    if sigma_sq < 1e-14:
        return max(min(V, H) - F, 0.0)
    sigma_ = sqrt(sigma_sq)

    sqrtT = sqrt(tau)
    gamma1 = (log((H + eps)/(F + eps)) +
              (r - 0.5*sigmaH**2)*tau) / (sigmaH*sqrtT)
    gamma2 = (log((V + eps)/(F + eps)) +
              (r - 0.5*sigmaV**2)*tau) / (sigmaV*sqrtT)

    y1 = (log((V + eps)/(H + eps)) - 0.5*sigma_sq*sqrtT) / (sigma_*sqrtT)
    y2 = (log((H + eps)/(V + eps)) - 0.5*sigma_sq*sqrtT) / (sigma_*sqrtT)

    theta1 = (rhoVH*sigmaV - sigmaH) / sigma_
    theta2 = (rhoVH*sigmaH - sigmaV) / sigma_

    disc = exp(-r*tau)

    term1 = H * bivariate_normal_cdf(gamma1 + sigmaH*sqrtT, y1, theta1)
    term2 = V * bivariate_normal_cdf(gamma2 + sigmaV*sqrtT, y2, theta2)
    term3 = F * disc * bivariate_normal_cdf(gamma1, gamma2, rhoVH)

    M_val = term1 + term2 - term3
    return M_val


def bs_option_price(spot, strike, r, sigma, T, t, option_type="call"):
    eps = 1e-14
    tau = T - t
    if tau < eps:
        if option_type.lower() == "call":
            return max(spot - strike, 0.0)
        elif option_type.lower() == "put":
            return max(strike - spot, 0.0)
    if spot < eps:
        if option_type.lower() == "call":
            return 0.0
        elif option_type.lower() == "put":
            return strike * exp(-r * tau)

    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2)
          * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option_type.lower() == "call":
        return spot * norm.cdf(d1) - strike * exp(-r * tau) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return strike * exp(-r * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def call_on_maximum(V, H, F, r, sigmaV, sigmaH, rhoVH, T, t):
    cV = bs_option_price(V, F, r, sigmaV, T, t)
    cH = bs_option_price(H, F, r, sigmaH, T, t)
    cMin = call_on_minimum(V, H, F, r, sigmaV, sigmaH, rhoVH, T, t)
    return cV + cH - cMin


def max_call_option_cube_stulz(
    Smax1=200, Smax2=200,
    dS1=2.0, dS2=2.0,
    T=1.0, r=0.05,
    sigma1=0.2, sigma2=0.25, rho=0.3,
    K=100,
    Nt=10
):
    N1 = int(Smax1/dS1)
    N2 = int(Smax2/dS2)
    S1_grid = np.linspace(0, Smax1, N1)
    S2_grid = np.linspace(0, Smax2, N2)

    t_grid = np.linspace(0, T, Nt+1)

    U = np.zeros((N1, N2, Nt+1))

    for i in range(N1):
        s1 = S1_grid[i]
        for j in range(N2):
            s2 = S2_grid[j]
            for k in range(Nt+1):
                t_val = t_grid[k]
                U[i, j, k] = call_on_maximum(
                    V=s1, H=s2, F=K,
                    r=r, sigmaV=sigma1, sigmaH=sigma2, rhoVH=rho,
                    T=T, t=t_val
                )

    return S1_grid, S2_grid, t_grid, U


def bs_price_cube(t, s, sigma, r, K, T, option_type="call"):
    prices = np.zeros((len(t), len(s)))
    for tt in range(len(t)):
        for ss in range(len(s)):
            prices[tt, ss] = bs_option_price(
                s[ss], K, r, sigma, T, t[tt], option_type)
    return prices
