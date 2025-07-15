import numpy as np
from derpinns.sampling import random_samples, scale_samples


class OptionParameters:
    '''
        Wrapper class with all required parameters. Given a set of parameters, it also computes the domain of each space variable.
    '''

    def __init__(self, n_assets, tau, sigma, rho, r, strike, payoff):
        self.n_assets = n_assets
        self.n_dim = n_assets + 1
        self.x_min = np.log(1 / strike)
        self.x_max = np.log(4)
        self.tau = tau
        self.sigma = sigma
        self.rho = rho
        self.r = r
        self.strike = strike
        self.payoff = payoff

    def domain_ranges(self, tau=None, fixed_asset_id=None, fixed_asset_value=None):
        # If no time is provided, use self.tau.
        if tau is None:
            tau = self.tau
        ranges = []
        for i in range(self.n_assets):
            if fixed_asset_id is not None and i == fixed_asset_id:
                ranges.append((fixed_asset_value, fixed_asset_value))
            else:
                ranges.append((self.x_min, self.x_max))
        ranges.append((0, tau))
        return ranges


def payoff(x):
    # Compute the option payoff (example: a call on the maximum of assets).
    x = np.max(np.exp(x), axis=1)
    payoff_values = np.maximum(x - 1, 0)
    return payoff_values.reshape([-1, 1])


def generate_collocations(n_samples, params: OptionParameters, mask_col, initial_condition=False,
                          fixed_asset_id=None, fixed_asset_value=None, sampler='pseudo', seed=None):
    '''
    Generates collocation points, their labels, and a binary mask.

    The mask is a matrix of shape (n_samples, num_loss_types) where:
         num_loss_types = 2 + 2 * params.n_assets.

    The mask column is specified by mask_col:
         - 0 for interior collocations.
         - 1 for initial condition collocations.
         - For asset boundaries:
               top boundary for asset i: mask_col = 2 + 2*i 
               bottom boundary for asset i: mask_col = 2 + 2*i + 1
    '''
    samples = random_samples(n_samples, params.n_dim, sampler, seed=seed)

    if initial_condition:
        # For the initial condition, time is 0.
        ranges = params.domain_ranges(tau=0)
        x = scale_samples(samples, ranges)
        y = params.payoff(x)
    elif fixed_asset_id is not None:
        # For boundary conditions, fix one asset variable.
        ranges = params.domain_ranges(fixed_asset_id=fixed_asset_id,
                                      fixed_asset_value=fixed_asset_value)
        x = scale_samples(samples, ranges)
        y = np.zeros((n_samples, 1))
    else:
        ranges = params.domain_ranges()
        x = scale_samples(samples, ranges)
        y = np.zeros((n_samples, 1))

    num_loss_types = 2 + 2 * params.n_assets
    mask = np.zeros((n_samples, num_loss_types), dtype=int)
    mask[:, mask_col] = 1

    return x, y, mask


def concat_datasets(dataset1, dataset2):
    '''
    Concatenates two datasets. Each dataset is a tuple (x, y, mask).
    '''
    x1, y1, mask1 = dataset1
    x2, y2, mask2 = dataset2
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    mask = np.concatenate((mask1, mask2), axis=0)
    return x, y, mask


def generate_dataset(params: OptionParameters, interior_samples, initial_samples, boundary_samples, sampler='pseudo', seed=None):
    '''
        Generates all the samples required to train the PINN (top, bottom, initial condition and interior samples).
    '''
    # Generate interior collocations (mask column 0).
    interior_points = generate_collocations(
        n_samples=interior_samples,
        params=params,
        mask_col=0,
        sampler=sampler,
        seed=seed
    )

    # Generate initial condition collocations (mask column 1) with time t = 0.
    initial_condition_points = generate_collocations(
        n_samples=initial_samples,
        params=params,
        mask_col=1,
        initial_condition=True,
        sampler=sampler,
        seed=seed
    )

    dataset = concat_datasets(interior_points, initial_condition_points)

    if boundary_samples > 0:
        for i in range(params.n_assets):
            # Top boundary for asset i (mask column 2 + 2*i).
            top_boundary_points = generate_collocations(
                n_samples=boundary_samples,
                params=params,
                mask_col=2 + 2 * i,
                fixed_asset_id=i,
                fixed_asset_value=params.x_max,
                sampler=sampler,
                seed=seed
            )
            # Bottom boundary for asset i (mask column 2 + 2*i + 1).
            bottom_boundary_points = generate_collocations(
                n_samples=boundary_samples,
                params=params,
                mask_col=2 + 2 * i + 1,
                fixed_asset_id=i,
                fixed_asset_value=params.x_min,
                sampler=sampler,
                seed=seed
            )

            dataset = concat_datasets(dataset, top_boundary_points)
            dataset = concat_datasets(dataset, bottom_boundary_points)

    x, y, mask = dataset
    return x, y, mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_assets = 2
    interior_samples = 10
    initial_samples = 10
    boundary_samples = 10

    tau = 1
    sigma = 0.2
    rho = 0.0
    r = 0.05
    strike = 60

    params = OptionParameters(n_assets, tau, sigma, rho, r, strike, payoff)
    x, y, mask = generate_dataset(params, interior_samples=interior_samples,
                                  initial_samples=initial_samples, boundary_samples=boundary_samples, sampler='Halton')

    print("Generated dataset:")
    print("x:", x.shape)
    print("y:", y.shape)
    print("mask:", mask.shape)
    # Optionally, print a few rows of the mask.
    print("Sample mask rows:")
    print(mask[:, 1])
    interior_mask = mask[:, 0] == 1
    interior_points = x[interior_mask]
    print(interior_mask, interior_points)
    plt.scatter(interior_points[:, 0], interior_points[:, 1])
    plt.show()
