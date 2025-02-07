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

- Gardiner, Handbook of stochastic methods, 2002
- Kampen, Stochastic processes in physics and chemistry, 2007
- Kampen, Ito versus Stratonovich, J. Stat. Phys., 1981
- Higham, An algorithmic introduction to numerical simulation of stochastic differential equations, SIAM Review, 2001
- Pechenik and Levine, PRE, 1999
- Dornic, Chate, and Munoz, PRL, 2005
- Cox and Matthews, J. Comp. Phys., 2002
