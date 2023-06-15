from dataclasses import dataclass
import numpy as np
from scipy.signal import welch
from typing import Optional

from rcsj_sde.utils import hbar_over_2e, v2_to_dbm
from rcsj_sde.junction import JosephsonJunction

@dataclass
class SimuResult():
    """
    Stores the result of an RCSJ SDE simulation. 
    
    Attributes
    ----------
    jj : JosephsonJunction
        Josephson junction instance used for the simulation (stores the junction parameters).
    I_DC_range: (N_I,) np.ndarray
        DC current points over which the simulation was carried out.
    tau : (N_tau,) np.ndarray
        Normalized time points
    sol_stored : (N_I, N_tau, 2) np.ndarray
        Stores phi and phidot, as a function of DC current and tau. Last index: 0 for phi and 1 for phidot.
    cutoff_threshold_idx : int
        Cutoff threshold index used for cutting off the transient period before
        calculating the PSD or the time-averaged voltage on the junction. By default 0.
    tau_max : int
        Length of the simulation in normalized time units.
    N_tau : int
        Number of time points of the simulation.
    plot_title : str
        String containing the most relevant parameters, convenient for plot titles.
    V : (N_I,) np.ndarray
        Time-averaged junction voltage as a function of DC current bias
    xf : (N_f,) np.ndarray
        Frequency axis for the power spectral density (PSD).
    psd : (N_I, N_f) np.ndarray
        Power spectral density of the junction voltage in linear units.
    psd_dbm : (N_I, N_f) np.ndarray
        Power spectral density of the junction voltage in logarithmic units (dBm).
    """

    jj: JosephsonJunction
    I_DC_range: np.ndarray
    tau: np.ndarray
    sol_stored: np.ndarray
    cutoff_threshold_idx: Optional[int] = 0
    
    def __post_init__(self):
        
        self.tau_max = int(np.max(self.tau))
        self.N_tau = len(self.tau)

        self.generate_title()
        self.calculate_psd() 
        self.calc_v()

    @classmethod
    def load(cls, fname: str)  -> "SimuResult":
        """
        Load a SimuResult from file.

        Parameters
        ----------
        fname : str
            File name used for loading (with or without extension). If provided without extension, .npz is added.

        Returns
        -------
        SimuResult
            SimuResult stored in fname
        """
        if not fname.endswith(".npz"):
            fname += ".npz"
            
        with np.load(fname) as data:
            jj = JosephsonJunction.from_json((data["jj"][0]))
            I_DC_range = data["I_DC_range"]
            tau = data["tau"]
            sol_stored = data["sol_stored"]
            cutoff_threshold_idx = data["cutoff_threshold_idx"]

        simu = cls(jj, I_DC_range, tau, sol_stored, cutoff_threshold_idx)
            
        return simu

    def save(self, fname: str) -> None:
        """
        Save a SimuResult to file

        Parameters
        ----------
        fname : str
            Filename used for saving (with or without extension). If provided without extension, .npz is added.
        
        Returns
        -------
        None
        """
        if not fname.endswith(".npz"):
            fname += ".npz"
            
        np.savez_compressed(fname,
                            jj=np.array([self.jj.to_json()]),
                            I_DC_range=self.I_DC_range,
                            tau=self.tau,
                            sol_stored=self.sol_stored,
                            cutoff_threshold_idx=np.array([self.cutoff_threshold_idx]),
                            )
            
    def export(self, fname: str) -> None:
        """
        Export the SimuResult to 4 seperate ASCII files:
        
        1) Current bias, junction voltage
        2) PSD in dBm scale
        3) PSD in linear scale
        4) Frequency values for the PSD

        Parameters
        ----------
        fname : str
            File name without extension
            
        Returns
        -------
        None
        """
        delimiter = ','
        header = f"""{str(self.jj)} 
                 tau_max = {self.tau_max:d}, tau_points = {self.N_tau:d}"""
        
        header_VI = header + f"\nI_DC, Voltage"

        np.savetxt(fname + "_VI.txt",
                   np.vstack([self.I_DC_range, self.V]).T,
                   delimiter=delimiter,
                   header=header_VI)
        
        np.savetxt(fname + "_PSD_dB.txt", self.psd_dbm, delimiter=delimiter, header=header)
        np.savetxt(fname + "_PSD.txt", self.psd, delimiter=delimiter, header=header)
        np.savetxt(fname + "_f.txt", self.xf, delimiter=delimiter, header=header)

    def generate_title(self) -> None:
        """
        Generate title string to be used in plots
                    
        Returns
        -------
        None
        """
        self.plot_title = (f"R={(self.jj.R):0.1f}, "
                           f"C={(self.jj.C):0.1e}, "
                           f"I_c={(self.jj.Ic):0.1e}, "
                           f"a={(self.jj.a):0.4g}, " 
                           f"b={(self.jj.b):0.4g}, "
                           f"T={(self.jj.T):0.2g}")
        
    def calc_v(self) -> None:
        """
        Calculate the DC junction voltage from phidot
        
        Returns
        -------
        None
        """
        I_points = len(self.I_DC_range)
        V = np.zeros(I_points);

        for k in range(I_points):
            phidot = self.sol_stored[k,self.cutoff_threshold_idx:,1]
            V[k] = hbar_over_2e * np.mean(phidot)/self.jj.t_c;

        self.V = V

    def calculate_psd(self, nseg=4, window='hann') -> None:
        """
        Calculate the power spectral density (PSD) of the junction voltage
        from phidot

        Parameters
        ----------
        nseg : int, optional
            Number of segments used by the Welch algorithm, by default 4
        window : str, optional
            Window size used by the Welch algorithm, by default 'hann'
            
        Returns
        -------
        None
        """
        n_signal = self.N_tau // nseg
        dtau = self.tau[1] - self.tau[0]
        dt = dtau * self.jj.t_c

        V_signal = hbar_over_2e * self.sol_stored[:,self.cutoff_threshold_idx:,1] / self.jj.t_c
        xf, psd = welch(V_signal,
                        fs=1.0/dt,
                        window=window,
                        nperseg=n_signal,
                        detrend="linear",
                        axis=1) # FFT of phidot

        self.xf, self.psd =  xf, psd
        self.psd_dbm = v2_to_dbm(psd) # dBm/Hz
