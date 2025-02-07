# StochasticProcesses.jl
Numerical methods for stochastic processes

# Getting started

> [!IMPORTANT]
> This is not a Julia package. You cannot install it with `add StochasticProcesses`.

## Prerequisites

You will need a working installation of Julia, jupyterlab, jupytext, and IJulia to generate
and run the notebooks. If you can run Julia notebooks on your machine, proceed to the next step.

- Install Julia. Using [juliaup](https://github.com/JuliaLang/juliaup) is recommended.
- Install jupyterlab and jupytext using anaconda or any other way you prefer.
- Install IJulia using Julia package manager.

## Running the notebooks

- Clone/download the repository.
- Install the Julia dependencies by activating the project and then instantiating it.
- The notebooks are converted and stored under `jl/` folder as plain `.jl` files using jupytext. To recreate the notebooks from these files run `make notebooks` and then move it to the base directory of the repository. You have to manually move them to avoid overwriting any notebooks you have previously generated.
-  If you do not have `make`, you can convert them directly by running `jupytext --to ipynb filename.jl`.
- Now you can run jupyterlab and start running the notebooks. 

# Notation

## General
- t: Time (usually an array of times)
- Δt: Time step
- tmax: Maximum time
- N: Number of time steps, calculated as tmax / Δt
- nens: Ensemble size (number of realizations)
- df: DataFrame
- pars: Set of parameters (usually a named tuple)

## Random Processes
- W: Wiener process or Brownian motion
- X: A general random process
- Xan: Analytical value of X
- ρ: A random process that cannot be negative, such as a density


# References

- Gardiner, Crispin W., **Handbook of Stochastic Methods: For Physics, Chemistry and the Natural Sciences** (2002).
- Van Kampen, N. G., **Stochastic Processes in Physics and Chemistry** (2007).
- Øksendal, B. K., **Stochastic Differential Equations: An Introduction with Applications** (2007).
- Van Kampen, N. G., **Itô versus Stratonovich**. [Journal of Statistical Physics 24(1) (1981)](https://doi.org/10.1007/BF01007642).
- Higham, Desmond J., **An Algorithmic Introduction to Numerical Simulation of Stochastic Differential Equations**. [SIAM Review 43(3) (2001)](https://doi.org/10.1137/S0036144500378302).
- Pechenik, Leonid, and Herbert Levine, **Interfacial Velocity Corrections Due to Multiplicative Noise**. [Physical Review E 59(4) (1999)](https://doi.org/10.1103/PhysRevE.59.3893).
- Dornic, Ivan, Hugues Chaté, and Miguel A. Muñoz, **Integration of Langevin Equations with Multiplicative Noise and the Viability of Field Theories for Absorbing Phase Transitions**. [Physical Review Letters 94(10) (2005)](https://doi.org/10.1103/PhysRevLett.94.100601).
- Cox, S.M., and P.C. Matthews. **Exponential Time Differencing for Stiff Systems**. [Journal of Computational Physics 176(2) (2002)](https://doi.org/10.1006/jcph.2002.6995).

