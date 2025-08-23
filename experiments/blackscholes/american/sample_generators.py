from scipy.stats import qmc
import numpy as np


def payoff_call(S, K):
    return np.maximum(S - K, 0)


def payoff_put(S, K):
    return np.maximum(K - S, 0)


def final_condition(params, S_array):
    # Final boundary (t=T) payoff depends on call or put
    strike = params['strike']
    option_type = params.get('option_type')
    if option_type == 'put':
        return payoff_put(S_array, strike)
    else:
        return payoff_call(S_array, strike)


def top_boundary_condition(params, t_array):
    """
    As S -> Smax (large S), for calls: V ~ S - K*exp(-r*(T-t))
    for puts: V ~ 0

    Since Smax is a large but finite number, we can approximate:
    - Call option top boundary: V_top(S=Smax,t) ≈ Smax - K*exp(-r*(T-t))
    - Put option top boundary: V_top(S=Smax,t) ≈ 0
    """
    r = params['r']
    strike = params['strike']
    T = params['maturity']
    Smax = params['Smax']
    option_type = params.get('option_type')
    exercise_style = params.get('exercise_style')
    if option_type == 'call':
        if exercise_style == 'european':
            return Smax - strike * np.exp(-r*(T - t_array))
        else:
            # American call: can exercise immediately
            return (Smax - strike)*np.ones_like(t_array)
        
    else:
        return np.zeros_like(t_array)


def bottom_boundary_condition(params, t_array):
    r = params['r']
    strike = params['strike']
    T = params['maturity']
    option_type = params.get('option_type')
    # You might also want a flag for "american" vs. "european," e.g.
    exercise_style = params.get('exercise_style')
    if option_type == 'call':
        # For a call, V(0,t) ~ 0 for both European and American
        return np.zeros_like(t_array)
    else:
        # Put:
        if exercise_style == 'european':
            # European put => must wait until expiry => discounted boundary
            return strike * np.exp(-r * (T - t_array))
        else:
            # American put => can exercise immediately => no discount
            return np.full_like(t_array, strike)


def generate_sobol_samples(sample_size, d):
    """
    Generate Sobol sequence samples.
    """
    sobol_engine = qmc.Sobol(d=d, scramble=True)
    return sobol_engine.random(sample_size)


def generate_sobol_samples(sample_size, d):
    """
    Generate Sobol sequence samples.
    """
    sobol_engine = qmc.Sobol(d=d, scramble=True)
    return sobol_engine.random(sample_size)


def generate_dataset_strategy(params,  method='uniform'):
    """
    Strategy 3:
    - Generate samples for PDE, boundaries, and final condition directly in the domain of S and t.
    - The number of samples for each is defined in `sample_counts`.
    - Supports Sobol and Chebyshev-Gauss-Lobatto sampling methods.

    Parameters:
    - params: Dictionary containing problem parameters.
    - sample_counts: Dictionary specifying sample counts, e.g.,
      {"pde": 200, "bottom_boundary": 50, "top_boundary": 50, "final_boundary": 100}.
    - method: Sampling method ('uniform', 'sobol', 'chebyshev_gauss_lobatto').

    Returns:
    - all_inputs: Combined inputs (S, t) for all samples.
    - all_outputs: Combined outputs (V) for all samples.
    - all_tags: Tags identifying sample types (0 for PDE, 1 for boundaries).
    """
    Smin = params['Smin']
    Smax = params['Smax']
    T = params['maturity']

    # Extract sample counts from the dictionary
    n_pde = params.get("pde", 0)
    n_bottom_boundary = params.get("bottom_boundary", 0)
    n_top_boundary = params.get("top_boundary", 0)
    n_final_boundary = params.get("final_boundary", 0)

    all_inputs = []
    all_outputs = []
    all_tags = []

    def sample_points(n, a, b):
        if method == 'sobol':
            sobol_samples = generate_sobol_samples(n, d=1)
            return a + (b - a) * sobol_samples.flatten()
        elif method == 'chebyshev_gauss_lobatto':
            return generate_chebyshev_gauss_lobatto_points(n, a, b)
        elif method == 'uniform':
            return np.random.uniform(a, b, n)
        else:
            raise ValueError(f"Invalid sampling method: {method}")

    def sample_points_2d(n, x_range, y_range):
        """
        Generate 2D points with Sobol, uniformly, or Chebyshev-Gauss-Lobatto.
        """
        if method == 'sobol':
            sobol_samples = generate_sobol_samples(n, d=2)
            x = x_range[0] + (x_range[1] - x_range[0]) * sobol_samples[:, 0]
            y = y_range[0] + (y_range[1] - y_range[0]) * sobol_samples[:, 1]
        elif method == 'uniform':
            x = np.random.uniform(*x_range, n)
            y = np.random.uniform(*y_range, n)
        elif method == 'chebyshev_gauss_lobatto':
            x = generate_chebyshev_gauss_lobatto_points(n, *x_range)
            y = generate_chebyshev_gauss_lobatto_points(n, *y_range)
        else:
            raise ValueError(f"Invalid sampling method: {method}")
        return x, y

    # Generate PDE points
    if n_pde > 0:
        S_pde, t_pde = sample_points_2d(n_pde, (Smin, Smax), (0, T))
        pde_inputs = np.column_stack([S_pde, t_pde])
        pde_outputs = np.zeros((n_pde, 1))  # PDE unknown, placeholder for V
        pde_tags = np.zeros((n_pde, 1))  # tag=0 for PDE
        all_inputs.append(pde_inputs)
        all_outputs.append(pde_outputs)
        all_tags.append(pde_tags)

    # Generate bottom boundary points (S=Smin)
    if n_bottom_boundary > 0:
        S_bottom = np.full(n_bottom_boundary, Smin)
        t_bottom = sample_points(n_bottom_boundary, 0, T)
        V_bottom = bottom_boundary_condition(params, t_bottom).reshape(-1, 1)
        bottom_inputs = np.column_stack([S_bottom, t_bottom])
        bottom_outputs = V_bottom  # Only V is needed
        bottom_tags = np.ones((n_bottom_boundary, 1))  # tag=1 for boundary
        all_inputs.append(bottom_inputs)
        all_outputs.append(bottom_outputs)
        all_tags.append(bottom_tags)

    # Generate top boundary points (S=Smax)
    if n_top_boundary > 0:
        S_top = np.full(n_top_boundary, Smax)
        t_top = sample_points(n_top_boundary, 0, T)
        V_top = top_boundary_condition(params, t_top).reshape(-1, 1)
        top_inputs = np.column_stack([S_top, t_top])
        top_outputs = V_top  # Only V is needed
        top_tags = np.ones((n_top_boundary, 1))  # tag=1 for boundary
        all_inputs.append(top_inputs)
        all_outputs.append(top_outputs)
        all_tags.append(top_tags)

    # Generate final boundary points (t=T)
    if n_final_boundary > 0:
        S_final = sample_points(n_final_boundary, Smin, Smax)
        t_final = np.full_like(S_final, T)
        V_final = final_condition(params, S_final).reshape(-1, 1)
        final_inputs = np.column_stack([S_final, t_final])
        final_outputs = V_final  # Only V is needed
        final_tags = np.ones((n_final_boundary, 1))  # tag=1 for boundary
        all_inputs.append(final_inputs)
        all_outputs.append(final_outputs)
        all_tags.append(final_tags)

    # Combine all datasets
    all_inputs = np.vstack(all_inputs)
    all_outputs = np.vstack(all_outputs)
    all_tags = np.vstack(all_tags)

    # Sort by time (t) and then by S
    sort_idx = np.lexsort((all_inputs[:, 0], all_inputs[:, 1]))
    all_inputs = all_inputs[sort_idx]
    all_outputs = all_outputs[sort_idx]
    all_tags = all_tags[sort_idx]

    return all_inputs, all_outputs, all_tags
