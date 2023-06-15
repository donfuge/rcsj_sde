import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
from operator import attrgetter
from typing import Optional, Sequence

from rcsj_sde.utils import linear_reference, ambegaokar_overdamped
from rcsj_sde.simuresult import SimuResult


def plotVI(simu: SimuResult, ax: Optional[plt.Axes]=None, legend: Optional[bool]=True) -> plt.Axes:
    """
    Plot the V-I curve from the simulation, with analytical references overlayed.

    Parameters
    ----------
    simu : SimuResult
        Simulation result to plot
    ax : plt.Axes, optional
        Matplotlib axes to use if given, by default None
    legend : bool, optional
        Puts legend on plot if True, by default True

    Returns
    -------
    plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,8))

    ax.plot(simu.I_DC_range/simu.jj.Ic, simu.V/(simu.jj.R*simu.jj.Ic), 'b', label='SDE result')
    ax.plot(simu.I_DC_range/simu.jj.Ic, ambegaokar_overdamped(simu.I_DC_range, simu.jj.Ic, simu.jj.R, 0)/(simu.jj.R*simu.jj.Ic), 'm--', label='overdamped ($T=0$)')
    ax.plot(simu.I_DC_range/simu.jj.Ic, ambegaokar_overdamped(simu.I_DC_range, simu.jj.Ic, simu.jj.R, simu.jj.T)/(simu.jj.R*simu.jj.Ic), 'k--', label='overdamped ($T>0$)')
    ax.plot(simu.I_DC_range/simu.jj.Ic, linear_reference(simu.I_DC_range, simu.jj.R)/(simu.jj.R*simu.jj.Ic), 'g--', label='simple resistor')

    if legend:
        ax.legend(loc='best')
    ax.set_xlabel(r'$I/I_c$')
    ax.set_ylabel(r'$V/(R I_c)$')
    ax.set_title(simu.plot_title)
    ax.grid()
    
    return ax

def inspect_sol(simu: SimuResult, I_idx: int) -> mpl_figure.Figure:
    """
    Inspect the solution of the SDE by plotting phi(tau), phidot(tau) and
    the PSD with both linear and logarithmic axes.

    Parameters
    ----------
    simu : SimuResult
        SimuResult to be inspected
    I_idx : int
        Bias current index (indexing I_DC_range) at which the solution is plotted

    Returns
    -------
    mpl_figure.Figure
        Matplotlib figure containing the plots.
    """
    sol = simu.sol_stored[I_idx]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=False, figsize=(14,8))
    
    ax1.plot(simu.tau, sol[:, 0], label=r'$\varphi (\tau)$')
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.axvline(simu.tau[simu.cutoff_threshold_idx])
    ax1.legend(loc='best')
    ax1.set_xlabel(r'$\tau$')
    ax1.set_ylabel(r'$\varphi$')
    ax1.set_title(f"i_dc = {simu.I_DC_range[I_idx]/simu.jj.Ic:.3f}")
    
    ax2.plot(simu.tau, sol[:, 1], label=r'$d\varphi (\tau) /d\tau$')
    ax2.axvline(simu.tau[simu.cutoff_threshold_idx])
    ax2.legend(loc='best')
    ax2.set_xlabel(r'$\tau$')
    ax2.set_ylabel(r'$d\varphi/d\tau$')
    ax2.autoscale(enable=True, axis='x', tight=True)

    ax3.plot(simu.xf,  simu.psd[I_idx,:], label='PSD')
    ax3.set_ylabel('PSD')
    # ax3.set_xlabel('Frequency [Hz]')
    ax3.autoscale(enable=True, axis='x', tight=True)

    ax4.semilogy(simu.xf,  simu.psd[I_idx,:], label='PSD')
    ax4.set_ylabel('PSD')
    ax4.set_xlabel(r'$f$ [Hz]')
    ax4.grid()
    ax4.autoscale(enable=True, axis='x', tight=True)

    plt.tight_layout()

    return fig

def plotFFTmap(simu: SimuResult,
               ax: Optional[plt.Axes]=None,
               legend: Optional[bool]=True,
               cbar: Optional[bool]=True,
               cbarloc: Optional[str]="right",
               vmin: Optional[float]=None,
               vmax: Optional[float]=None) -> plt.Axes:
    """
    Create a 2D plot of the PSD, x axis is frequency, y axis is voltage.

    Parameters
    ----------
    simu : SimuResult
        Simulation to be plotted
    ax : Optional[plt.Axes], optional
        Matplotlib axes used for plotting if provided, by default None
    legend : Optional[bool], optional
        Makes a legend if True, by default True
    cbar : Optional[bool], optional
        Makes a colorbar if True, by default True
    cbarloc : Optional[str], optional
        Location of the colorbar, by default "right"
    vmin : Optional[float], optional
        Minimum of the colormap values, by default None
    vmax : Optional[float], optional
        Maximum of the colormap values, by default None

    Returns
    -------
    plt.Axes
        Matplotlib axes containing the plot
    """
    # TODO settable threshold
    V_th = simu.jj.R*simu.jj.Ic/1e2 # voltage threshold for plotting
    V_start_idx = np.min(np.where(simu.V > V_th)) + 1
    xf_start_idx = 0

    K_J = 483597.8484e9 # (2e)/h, Josephson constant in Hz/V 
    
    annot_factors = [2, 1, 3.0/2, 0.5]
    periods = ["1", "2", "4/3", "4"]
    annot_labels = [f"${s}\\pi$" for s in periods] 
    V_annot = []
    aiis = []
    
    annot_cutoff_voltage = 8e-6

    for factor in annot_factors:
        V = simu.xf/(K_J*factor)
        V_annot.append(V)
        aiis.append(V > annot_cutoff_voltage)
    
    if ax is None:
        fig, ax = plt.subplots()

    if vmin is None:
        vmin=np.min(simu.psd_dbm[:])
    
    if vmax is None:
        vmax=np.max(simu.psd_dbm[:])

    f_unit = 1e9
    V_unit = 1e-6

    pm = ax.pcolormesh(simu.xf[xf_start_idx:]/f_unit,
                    simu.V[V_start_idx:]/V_unit,
                    simu.psd_dbm[V_start_idx:,xf_start_idx:],
                    vmin=vmin,
                    vmax=vmax,
                    shading='auto',
                    rasterized=True)

    ax.set_prop_cycle(color=['red', 'magenta', 'blue', 'black'])
    
    for V, aii, label in zip(V_annot, aiis, annot_labels):
        ax.plot(simu.xf[aii]/f_unit,
                 V[aii]/V_unit,
                 linewidth=0.5,
                 linestyle="--", 
                 label=label)
    
    if cbar: 
        plt.colorbar(pm, location=cbarloc, ax=ax)
        
    if legend:
        ax.legend(loc="lower right")

    ax.set_xlim([0, 10])
    ax.set_ylim([0, max(simu.V/V_unit)])

    ax.set_xlabel("$f$ [GHz]")
    ax.set_ylabel("$V_{dc}$ [uV]")
    ax.set_title(simu.plot_title)
    
    return ax
 
def overlay_cuts(simus: Sequence[SimuResult],
                 param: str,
                 paramvals: Sequence[float],
                 ax: Optional[plt.Axes]=None,
                 V_cut: Optional[int]=None,
                 legend: Optional[bool]=True,
                 reverse: Optional[bool]=True,
                 alpha: Optional[float]=1) -> plt.Axes:
    """
    Create a plot with overlaying PSD cuts as a function of frequency
    from different simulations at fixed voltage index.

    Parameters
    ----------
    simus : Sequence[SimuResult]
        SimuResult instances
    param : str
        Parameter which varies across the simulations, e.g. jj.C
    paramvals : Sequence[float]
        Values of the parameters to be included in the plot
    ax : Optional[plt.Axes], optional
        Matplotlib axes to use for plotting, by default None
    V_cut : Optional[int], optional
        Voltage index for the frequency cut, by default None
    legend : Optional[bool], optional
        Shows legend if True, by default True
    reverse : Optional[bool], optional
        Reverses the order of simus plotting if True, by default True
    alpha : Optional[float], optional
        Alpha of the line color in the plot, by default 1

    Returns
    -------
    plt.Axes
        Matplotlib axes containing the plot
    """
    f_unit = 1e9

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,5))

    r, c = simus[0].psd.shape

    paramstr = param.split(".")[-1]

    if reverse:
        simus = simus[::-1]
        
    simus_filtered = [s for s in simus if attrgetter(param)(s) in paramvals]
    colors = plt.cm.coolwarm(np.linspace(1,0,len(simus_filtered)))

    for i, simu in enumerate(simus_filtered):

        if V_cut is None:
            V_idx = r - 1
        else:
            V_idx = np.argmin(np.abs(simu.V - V_cut))

        paramval = attrgetter(param)(simu)
        ax.plot(simu.xf/f_unit, 
                 simu.psd_dbm[V_idx,:], 
                 alpha=alpha, 
                 label=f"{paramstr} = {paramval:0.2g}", 
                 color=colors[i])

    ax.set_xlabel("$f$ [GHz]")
    ax.set_ylabel("PSD [dBm/Hz]")
    
    if legend:
        ax.legend(loc="upper right", frameon=False)

    return ax
