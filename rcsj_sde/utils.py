import numpy as np
from scipy.integrate import quad
from scipy.special import iv  # modified Bessel function of the 1st kind
from typing import Union, Optional

# reduced Planck constant / (2*elementary charge) in SI units [weber]
hbar_over_2e = 3.29106e-16
# Boltzmann constant in SI units [joule/kelvin]
kB = 1.380649e-23


def overdamped_zerotemperature(I: Union[float, np.ndarray], I_c: float, R: float) -> Union[float, np.ndarray]:
    """
    Calculate the junction voltage in the overdamped limit for T=0.

    Parameters
    ----------
    I : Union[float, np.ndarray]
        Bias current
    I_c : float
        Critical current of the junction
    R : float
        Parallel resistance

    Returns
    -------
    np.ndarray or float
        Junction voltage, same shape as input I
    """
    V = np.real(I_c * R * np.emath.sqrt((I/I_c)**2-1))
    return V


def linear_reference(I: Union[float, np.ndarray], R: float) -> Union[float, np.ndarray]:
    """
    Calculate the voltage drop over resistor R at current I.

    Parameters
    ----------
    I : Union[float, np.ndarray]
        Current
    R : float
        Resistance

    Returns
    -------
    Union[float, np.ndarray]
        Voltage on resistor, same shape as input I
    """
    V = I*R
    return V

def ambegaokar_overdamped(I: Union[float, np.ndarray],
                          I_c: float,
                          R: float,
                          T: Optional[float] = None,
                          gamma0: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Calculate the junction voltage in the overdamped limit for arbitrary T>=0.
    
    See Eq. 3.4.21 in the textbook Applied Superconductivity by Prof. Dr. Rudolf Gross and  Dr. Achim Marx:

    https://www.wmi.badw.de/fileadmin/WMI/Lecturenotes/Applied_Superconductivity/AS_Chapter3.pdf

    Either provide T or gamma0. 

    Parameters
    ----------
    I : Union[float, np.ndarray]
        Current bias
    I_c : float
        Junction critical current
    R : float
        Parallel shunt resistor
    T : float, optional
        Temperature in kelvin, by default None
    gamma0 : float, optional
        Gamma parameter, by default None

    Returns
    -------
    Union[float, np.ndarray]
        Junction voltage, same shape as input I
    """

    def integrand(phi, i):
        # iv: modified Bessel function
        return np.exp(-i*gamma0*phi/2)*iv(0, gamma0*np.sin(phi/2))

    if T is None and gamma0 is None:
        raise ValueError("Either T or gamma0 must be provided")

    if T is not None and gamma0 is not None:
        raise ValueError(
            "Arguments T and gamma0 cannot be provided simultaneously")

    if T is not None:
        if T == 0:
            return overdamped_zerotemperature(I, I_c, R)
        elif T > 0:
            # 3.4.18 (flux quantum = hbar_over_2e*2*np.pi)
            gamma0 = hbar_over_2e*2*np.pi*I_c/(np.pi*kB*T)
        else:
            raise ValueError("Argument T cannot be negative")

    if isinstance(I, np.ndarray):
        ispan = I/I_c
        integral = np.zeros_like(ispan)
        for i_idx in range(len(ispan)):
            integral[i_idx] = quad(integrand, 0, 2*np.pi,
                                   args=(ispan[i_idx],))[0]
        x = ispan*np.pi*gamma0

    else:
        i = I/I_c
        integral = quad(integrand, 0, 2*np.pi, args=(i,))[0]
        x = i*np.pi*gamma0

    return 2*I_c*R/gamma0*(1 - np.exp(-x))*1/integral


def v2_to_dbm(psd: Union[float, np.ndarray], R: float = 50) -> Union[float, np.ndarray]:
    """
    Convert power spectral density from V^2/Hz to dBm/Hz.

    Parameters
    ----------
    psd : Union[float, np.ndarray]
        Power spectral density in V^2/Hz
    R : float, optional
        Resistance, by default 50

    Returns
    -------
    Union[float, np.ndarray]
        Power spectral density in dBm/Hz
    """
    return 10*np.log10(psd/R) + 30


def watt_to_dbm(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert power from watt to dBm

    Parameters
    ----------
    p : Union[float, np.ndarray]
        Power in watt

    Returns
    -------
    Union[float, np.ndarray]:
        Power in dBm
    """
    return 10*np.log10(p) + 30


def thermal_noise_voltage(T: Union[float, np.ndarray], R: float) -> Union[float, np.ndarray]:
    """
    Calculate the thermal noise voltage of a resistor per unit bandwidth in <V**2>/Hz

    Parameters
    ----------
    T : float
        Temperature in kelvin
    R : float
        Resistor value

    Returns
    -------
    Union[float, np.ndarray]
        Thermal noise voltage in <V**2>/Hz
    """
    return 4*kB*T*R

def thermal_noise_power(T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate the thermal noise power of a resistor per unit bandwidth in <P>/Hz.

    Parameters
    ----------
    T : float
        Temperature in kelvin

    Returns
    -------
    Union[float, np.ndarray]
        Thermal noise power in <P>/Hz
    """
    return 4*kB*T

    
def symmetrize(array: np.ndarray) -> np.ndarray:
    """
    Generate symmetrized array by mirroring, tailored for Shapiro plots.
    
    For a 1D array (e.g. bias current), it negates and flips the mirrored part.
    For a 2D array (e.g. voltage output of Shapiro simulation), it only flips
    along axis 1 to make the mirrored part.

    Parameters
    ----------
    array : np.ndarray
        1D or 2D array to be symmetrized

    Returns
    -------
    np.ndarray
        Symmetrized array

    Raises
    ------
    Exception
        If ndim > 2, raises an error.
    """
    if array.ndim == 1:
        array_sym = array[:-1]
        array_sym = np.concatenate([-array_sym[::-1], array_sym[1:]])
    elif array.ndim == 2:
        r, c = array.shape
        array_sym = np.zeros((r, 2*c-1))
        array_sym[:, c-1:] = array
        array_sym[:, :c] = np.fliplr(array)
    else:
        raise Exception("Only works for 1D and 2D arrays.")

    return array_sym
