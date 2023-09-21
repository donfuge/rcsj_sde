# Resistively and Capacitively Shunted Junction (RCSJ) model simulator with thermal noise 

Code repository accompanying our paper titled "AC Josephson effect in a gate-tunable Cd<sub>3</sub>As<sub>2</sub> nanowire superconducting weak link" [1,2].

## Outline

Our program simulates the phase dynamics of a Josephson junction under the external bias current $I(t)$, influenced by the thermal noise current $I_{F}(t)$. It solves the differential equation

$$
\frac{\hbar}{2e} C \frac{d^2 \varphi}{dt^2} + \frac{\hbar}{2e} \frac{1}{R} \frac{d\varphi}{dt} + I_c f(\varphi) + I_{F}(t) = I(t) 
$$

for the junction phase $\varphi(t)$. In the presence of the random current fluctuation term $I_{F}(t)$ this is a stochastic differential equation (SDE). We use Heun's method for the numerical integration, for details see the supplementary material of our paper.

The current-phase relation is represented by $f(\varphi) = a \sin(\varphi) + b \sin(\varphi/2)$, where $a$ and $b$ are the weights of the conventional and topological supercurrent term, respectively. Thus, the simulation of purely conventional, purely topological and mixed junctions is possible.

## Requirements

* python >= 3.7
* numpy
* scipy
* matplotlib
* numba
* tqdm

## Installation

Clone the repository:

```
git clone https://github.com/donfuge/rcsj_sde.git
```

And install in editable mode with `pip`:

```
pip install -e rcsj_sde
```

## Basic usage

The functions `simu.simulation_full()` and `simu.simulation_voltage()` run an experiment with ramped DC current bias. For each value of the DC bias, they solve the SDE, resulting in $\varphi(\tau)$ and $\dot\varphi(\tau)$. 

1) `simu.simulation_voltage()` only returns the time-averaged junction voltage as a function of the bias current. It enables the investigation of the V-I curve of the Josephson junction. This function is recommended to simulate Shapiro experiments, where both the DC current and the AC amplitude are swept, resulting in a potentially large dataset.

2) `simu.simulation_full()` returns the full solution, $\varphi(\tau)$ and $\dot\varphi(\tau)$ for each bias value, in the form of a `SimuResult` object. This enables the investigation of the phase dynamics at specific bias current values. It calculates the power spectral density of the Josephson radiation as well.

## Examples

1) See [examples/basic_example.ipynb](examples/basic_example.ipynb) for a V-I curve and Josephson radiation simulation.

2) See [examples/shapiro_example.ipynb](examples/shapiro_example.ipynb) for a Shapiro simulation.

3) In [examples/shapiro_topo_sweep.ipynb](examples/shapiro_topo_sweep.ipynb) you can find Shapiro maps for different current-phase relations, with varying $a, b$ parameters.

## Citation

If you use this code for a scientific publication, please cite our paper:

```
@article{PhysRevB.108.094514,
  title = {ac Josephson effect in a gate-tunable ${\mathrm{Cd}}_{3}{\mathrm{As}}_{2}$ nanowire superconducting weak link},
  author = {Haller, R. and Osterwalder, M. and F\"ul\"op, G. and Ridderbos, J. and Jung, M. and Sch\"onenberger, C.},
  journal = {Phys. Rev. B},
  volume = {108},
  issue = {9},
  pages = {094514},
  numpages = {8},
  year = {2023},
  month = {Sep},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.108.094514},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.108.094514}
}
```

## References

[1] [Our paper on arXiv](https://arxiv.org/abs/2305.19996)

[2] [Phys. Rev. B 108, 094514 (2023)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.094514)

