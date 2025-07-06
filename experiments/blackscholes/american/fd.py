import numpy as np

def option_value_fd_american(vol, int_rate, strike, expiration, option_type='put', nas=200):
    """
    Compute the American option price (put or call) using a forward-Euler
    finite difference method in 2D (S vs. t).

    Parameters
    ----------
    vol        : float
        Volatility (sigma).
    int_rate   : float
        Risk-free interest rate (r).
    strike     : float
        Strike price (K).
    expiration : float
        Time to expiration (T).
    option_type: str, 'put' or 'call'
        Type of the option to be priced.
    nas        : int
        Number of stock-price steps in [0, Smax].
        
    Returns
    -------
    v : ndarray of shape (nas+1, nts+1)
        v[i, k] = approximate option value at S_i and time step k.
        Note: k=0 corresponds to t=0 if we store time forward.
    s : ndarray of shape (nas+1,)
        Array of stock prices for each index i.
    t : ndarray of shape (nts+1,)
        Array of times for each index k, from 0 up to T.
    """
    # 1) Set up the grid
    Smax = 4.0 * strike  # large enough upper bound for S
    ds = Smax / nas
    s = np.linspace(1e-4, Smax, nas+1)
    
    # stability constraint for explicit Euler
    dt_est = 0.9 / (vol**2 * nas**2)
    nts = int(np.ceil(expiration / dt_est))
    dt = expiration / nts
    t_array = np.linspace(0.0, expiration, nts+1)

    # 2) Initialize payoff and FD array
    if option_type.lower() == 'put':
        payoff = np.maximum(strike - s, 0.0)  # payoff for a put
    elif option_type.lower() == 'call':
        payoff = np.maximum(s - strike, 0.0)  # payoff for a call
    else:
        raise ValueError("option_type must be 'put' or 'call'")
    
    v = np.zeros((nas+1, nts+1))
    # At time t=0, the American option can be exercised immediately
    v[:, 0] = payoff  

    # 3) Main time-stepping loop
    for k in range(1, nts+1):
        # interior points i=1..(nas-1)
        for i in range(1, nas):
            # first derivative
            delta = (v[i+1, k-1] - v[i-1, k-1]) / (2.0 * ds)
            # second derivative
            gamma = (v[i+1, k-1] - 2.0*v[i, k-1] + v[i-1, k-1]) / (ds**2)

            theta = (
                0.5 * vol**2 * s[i]**2 * gamma
                + int_rate * s[i] * delta
                - int_rate * v[i, k-1]
            )
            
            # Forward-Euler update
            v[i, k] = v[i, k-1] + dt * theta
        
        # 3a) Boundary conditions
        if option_type.lower() == 'put':
            # At S=0 for American put: immediate exercise payoff ~ K
            v[0, k] = strike
            # At S=Smax for American put: value ~ 0
            v[nas, k] = 0.0
        else:
            # For an American call:
            # At S=0, the call is worthless
            v[0, k] = 0.0
            # At S=Smax, approximate the value by (Smax - K), 
            # though for a large S this is a reasonable boundary
            v[nas, k] = (s[nas] - strike)
        
        # 3b) Early exercise constraint
        for i in range(nas+1):
            v[i, k] = max(v[i, k], payoff[i])

    return v, s, t_array