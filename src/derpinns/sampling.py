import numpy as np
import skopt
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="skopt"
)


def random_samples(n_samples, dimension, sampler="pseudo", seed=None):
    if sampler == "pseudo":
        return pseudorandom(n_samples, dimension, seed)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler, seed)
    raise ValueError("f{sampler} sampling is not available.")


def pseudorandom(n_samples, dimension, seed=None):
    np.random.seed(seed)
    return np.random.random(size=(n_samples, dimension))


def quasirandom(n_samples, dimension, sampler, seed=None):
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    skip = 0
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs()
    elif sampler == "Halton":
        # 1st point: [0, 0, ...]
        sampler = skopt.sampler.Halton(min_skip=1, max_skip=1)
    elif sampler == "Hammersley":
        # 1st point: [0, 0, ...]
        if dimension == 1:
            sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
        else:
            sampler = skopt.sampler.Hammersly()
            skip = 1
    elif sampler == "Sobol":
        # 1st point: [0, 0, ...], 2nd point: [0.5, 0.5, ...]
        sampler = skopt.sampler.Sobol(randomize=False)
        if dimension < 3:
            skip = 1
        else:
            skip = 2
    space = [(0.0, 1.0)] * dimension
    return np.asarray(
        sampler.generate(space, n_samples + skip, random_state=seed)[skip:]
    )


def scale_samples(samples, ranges):
    scaled_samples = []
    for i in range(samples.shape[1]):
        a, b = ranges[i]
        scaled_samples.append(a + (b - a) * samples[:, i])
    return np.array(scaled_samples).T


def residual_based_adaptive_sampling(res_f, n_samples, ranges, k, c, sampler="pseudo", seed=None):
    """
    Implementation of the residual-based adaptive sampling method.
    https://www.sciencedirect.com/science/article/pii/S0045782522006260?via%3Dihub
    """
    samples = random_samples(n_samples*10, len(ranges), sampler, seed=seed)
    scl_samples = scale_samples(samples, ranges)
    residual = res_f(scl_samples)

    err = np.abs(residual)
    err_eq = np.power(err, k) / np.power(err, k).mean() + c
    err_eq_normalized = err_eq / np.sum(err_eq)

    selected_ids = np.random.choice(
        a=len(scl_samples), size=n_samples, replace=False, p=err_eq_normalized)
    selected_samples = scl_samples[selected_ids, :]
    return selected_samples


if __name__ == "__main__":
    # Example usage of residual_based_adaptive_sampling
    import matplotlib.pyplot as plt

    def res_f(x):
        a = 10
        e = 2**(4*a)*x[:, 0]**a*(1-x[:, 0])**a*x[:, 1]**a*(1-x[:, 1])**a
        return e

    n_samples = 1000
    n_dims = 2
    ranges = [(0, 1), (0, 1)]
    k = 0.5
    c = 2
    samples = random_samples(n_samples, n_dims, sampler='Halton')
    scaled_samples = scale_samples(samples, ranges)
    selected_samples = residual_based_adaptive_sampling(
        res_f, n_samples, ranges, k, c, sampler='Halton')
    print("Selected Samples:\n", selected_samples)
    plt.scatter(scaled_samples[:, 0], scaled_samples[:, 1], label='Samples')
    plt.scatter(selected_samples[:, 0], selected_samples[:,
                1], label='Selected Samples', color='red')
    plt.legend()
    plt.show()
