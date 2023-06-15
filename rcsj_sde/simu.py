import numpy as np
from tqdm import tqdm

from rcsj_sde.rcsj import rcsj_solver
from rcsj_sde.simuresult import SimuResult
from rcsj_sde.utils import hbar_over_2e
from rcsj_sde.junction import JosephsonJunction


def simulation_voltage(jj: JosephsonJunction,
                       I_DC_range: np.ndarray,
                       I_ac: float = 0.0,
                       F_ac: float = 0.0,
                       tau_max: int = 2_000,
                       tau_points: int = 20_000,
                       progressbar: bool = True) -> np.ndarray:
    """
    Run a current bias simulation, return only the time-averaged junction voltage
    as a function of current bias.

    Parameters
    ----------
    jj : JosephsonJunction
        Josephson junction instance
    I_DC_range : np.ndarray
        DC current bias values in ampere (for example, an up ramp with increasing values of DC current)
    I_ac : float, optional
        Amplitude of the AC current bias in ampere (for Shapiro experiments), by default 0.0
    F_ac : float, optional
        Frequency of the AC current bias in hertz (if present, for Shapiro experiments), by default 0.0
    tau_max : int, optional
        Length of the time-domain solution of the SDE (in normalized units), by default 2000
    tau_points : int, optional
        Number of points for the solution of the SDE, by default 20000
    progressbar : bool, optional
        Shows the tqdm progress bar if True, by default True

    Returns
    -------
    (Ni,) np.ndarray 
        Time-averaged junction voltage
    """
    I_points = len(I_DC_range)

    i_ac = I_ac/jj.Ic  # AC current bias amplitude in normalized units
    f_ac = F_ac*jj.t_c  # frequency in normalized units
    i_dc_range = I_DC_range/jj.Ic  # DC current amplitude in normalized units

    # normalized time, tau = t/tau_J
    tau = np.linspace(0, tau_max, tau_points)
    
    # initial conditions for the first bias value: phi=0, phidot=0
    y0 = np.asarray([0, 0], dtype=np.float64)

    # reserve storage for output 
    V = np.zeros((I_points,))  # junction voltage

    # loop over current bias values
    for (k, i_dc) in enumerate(tqdm(i_dc_range, position=0, leave=True, disable=not progressbar)):
        sol = rcsj_solver(jj.epsilon, jj.beta, i_dc, jj.a, jj.b, y0, tau, i_ac, f_ac)

        V[k] = hbar_over_2e*np.mean(sol[:, 1])/jj.t_c

        # carry over the final phi, phidot as the initial condition for the next bias current
        reduced_phi = sol[-1, 0] % (2*np.pi)
        y0 = np.asarray([reduced_phi, sol[-1, 1]], dtype=y0.dtype)

    return V


def simulation_full(jj: JosephsonJunction,
               I_DC_range: np.ndarray,
               I_ac: float = 0.0,
               F_ac: float = 0.0,
               tau_max: int = 2_000,
               tau_points: int = 20_000,
               progressbar: bool = True) -> SimuResult:
    """
    Run a current bias simulation and return a SimuResult, containing phi(tau) and phidot(tau) for
    all bias current values.

    Parameters
    ----------
    jj : JosephsonJunction
        Josephson junction instance
    I_DC_range : np.ndarray
        DC current bias values in ampere (for example, an up ramp with increasing values of DC current)
    I_ac : float, optional
        Amplitude of the AC current bias in ampere (for Shapiro experiments), by default 0.0
    F_ac : float, optional
        Frequency of the AC current bias in hertz (if present, for Shapiro experiments), by default 0.0
    tau_max : int, optional
        Length of the time-domain solution of the SDE (in normalized units), by default 2000
    tau_points : int, optional
        Number of points for the solution of the SDE, by default 20000
    progressbar : bool, optional
        If True, it shows the tqdm progress bar, by default True

    Returns
    -------
    SimuResult
        SimuResult instance containing phi(tau) and phidot(tau).
    """

    I_points = len(I_DC_range)

    i_ac = I_ac/jj.Ic  # AC current bias amplitude in normalized units
    f_ac = F_ac*jj.t_c  # frequency in normalized units
    i_dc_range = I_DC_range/jj.Ic  # DC current amplitude in normalized units

    # normalized time, tau = t/tau_J
    tau = np.linspace(0, tau_max, tau_points)
    
    # initial conditions for the first bias value: phi=0, phidot=0
    y0 = np.asarray([0, 0], dtype=np.float64)

    # reserve storage for output 
    sol_stored = np.zeros((I_points, tau_points, 2)) # last axis dim: 2, for phi and phidot

    # loop over current bias values
    for (k, i_dc) in enumerate(tqdm(i_dc_range, position=0, leave=True, disable=not progressbar)):
        sol = rcsj_solver(jj.epsilon, jj.beta, i_dc, jj.a, jj.b, y0, tau, i_ac, f_ac)

        sol_stored[k, :] = sol

        # carry over the final phi, phidot as the initial condition for the next bias current
        reduced_phi = sol[-1, 0] % (2*np.pi)
        y0 = np.asarray([reduced_phi, sol[-1, 1]], dtype=y0.dtype)

    return SimuResult(jj, I_DC_range, tau, sol_stored)
