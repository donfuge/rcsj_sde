import numpy as np
from math import sin
import numba


@numba.njit
def rng_wrapper(h, N):
    """
    Generate random numbers with Gaussian distribution.
    Wrapper needed for numba, see
    https://numba-how-to.readthedocs.io/en/latest/numpy.html#using-the-new-1-17-numpy-random-generator

    Parameters
    ----------
    h : float
        Time step size
    N : int
        Number of random numbers

    Returns
    -------
    np.ndarray
        N independent Gaussian-distributed N(0, sqrt(h)) random numbers
    """
    with numba.objmode(dW="float64[:]"):
        generator = np.random.default_rng()
        dW = generator.normal(0.0, np.sqrt(h), (N,))
    return dW

@numba.njit
def rcsj_solver(epsilon: float,
                beta: float,
                i_dc: float,
                a: float,
                b: float,
                y0: np.ndarray,
                tspan: np.ndarray,
                i_ac: float=0.0,
                f_ac: float=0.0) -> np.ndarray:
    """
    Solve the RCSJ SDE with Heun's method. The RCSJ SDE is formulated
    in normalized units.
    
    For a reference on Heun's method, see:
    Numerical Treatment of Stochastic Differential Equations
    W. Rümelin
    SIAM Journal on Numerical Analysis
    Vol. 19, No. 3 (Jun., 1982), pp. 604-613 
    https://www.jstor.org/stable/2156972?origin=JSTOR-pdf

    Parameters
    ----------
    epsilon : float
        Noise term, epsilon=sigma/beta.
    beta : float
        Stewart-McCumber parameter
    i_dc : float
        DC bias current (in normalized units)
    a : float
        Prefactor of trivial term in the current-phase relation, a*sin(phi)
    b : float
        Prefactor of topological term in the current-phase relation, b*sin(phi/2)         
    y0 : np.ndarray
        Initial conditions, phi and phidot
    tspan : np.ndarray
        Time span of the solution (in normalized units)
    i_ac : float
        Amplitude of the AC bias current (in normalized units)
    f_ac : float
        Frequency of the AC bias current (in normalized units)
        
    Returns
    -------
    np.ndarray
        Solution of the SDE, phi and phidot
    """
    d = len(y0)
    N = len(tspan)
    h = tspan[1] - tspan[0]
    dW = rng_wrapper(h, N)
    y = np.zeros((N, d), dtype=y0.dtype)
    f1 = np.zeros((2,), dtype=y0.dtype)
    f2 = np.zeros_like(f1)
    y[0] = y0;

    for n in range(0, N-1):
        yn = y[n] 
        itot = i_dc + i_ac*sin(2*np.pi*f_ac*n*h)
        f1[0] = yn[1]*h
        f1[1] = 1/beta*(itot - (a*sin(yn[0]) + b*sin(yn[0]/2)) - yn[1])*h 
        k1 = yn + f1 
        f2[0] = k1[1]*h
        f2[1] = 1/beta*(itot - (a*sin(k1[0]) + b*sin(k1[0]/2)) - k1[1])*h 
        Yn1 = yn + 0.5*(f1 + f2)
        Yn1[1] += epsilon*dW[n] # add noise
        y[n+1] = Yn1

    return y

