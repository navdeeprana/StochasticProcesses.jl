# StochasticProcesses.jl
Numerical methods for stochastic processes

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