# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Julia 1.10.8
#     language: julia
#     name: julia-1.10
# ---

# %%
# Imports and setup
import Pkg;
Pkg.activate(".");
using Revise, Printf, CairoMakie, DataFrames, StatsBase, Random, UnPack
includet("src/plotting.jl")
includet("src/brownian.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

# %% [markdown]
# # Riemann integrals

# %%
function riemann_integral(f::F1, xpoint::F2, xs, xf, N) where {F1,F2}
    Δx = (xf - xs) / N
    sum = 0
    for i in 1:N
        xi = xpoint(xs, i, Δx)
        sum += f(xi)
    end
    return Δx * sum
end

leftpoint(xs, i, Δx) = xs + (i - 1) * Δx
riemann_integral_left(f, xs, xf, N) = riemann_integral(f, leftpoint, xs, xf, N)

midpoint(xs, i, Δx) = xs + (i - 1) * Δx + Δx / 2
riemann_integral_mid(f, xs, xf, N) = riemann_integral(f, midpoint, xs, xf, N)

# %%
f(x) = 3x^2
@show riemann_integral_left(f, 0, 1, 20);
@show riemann_integral_mid(f, 0, 1, 20);

# %%
N = 2 .^ (4:10)
Δs1 = [abs(riemann_integral_left(f, 0, 1, Ni) - 1) for Ni in N]
Δs2 = [abs(riemann_integral_mid(f, 0, 1, Ni) - 1) for Ni in N]

fig, ax = figax(h = 5, xscale = log2, yscale = log2, xlabel = "N", ylabel = "Δs")
scatter!(ax, N, Δs1, label = "Left")
lines!(ax, N, power_law(N, -1, Δs1[1]), color = :black)
scatter!(ax, N, Δs2, label = "Midpoint")
lines!(ax, N, power_law(N, -2, Δs2[1]), color = :black)
axislegend(ax, position = :lb)
fig

# %% [markdown]
# ## HW : Find analytical expression for the left and mid rules.
# ## HW : Implement the right Riemann integral and find out how it converges.

# %% [markdown]
# # Stochastic integrals

# %%
# Stochastic integrals for the Wiener process W
function ito_integral(W)
    N, s = length(W), 0.0
    for i in 1:(N-1)
        s += W[i] * (W[i+1] - W[i])
    end
    return s
end

function strato_integral(W, V)
    N, s = length(W), 0.0
    for i in 1:(N-1)
        s += (0.5 * (W[i+1] + W[i]) + V[i]) * (W[i+1] - W[i])
    end
    return s
end

# One line variants of the above functions
@inbounds ito_oneliner(W) = sum(W[i] * (W[i+1] - W[i]) for i in 1:(length(W)-1))

@inbounds strato_oneliner(W, V) = sum((0.5 * (W[i+1] + W[i]) + V[i]) * (W[i+1] - W[i]) for i in 1:(length(W)-1))

percentage_error(a, b) = 100 * abs((a - b) / a)

# %% [markdown]
# ## HW : Prove the expression for $W((t_j + t_{j+1})/2)$

# %%
tmax, Δt = 1, 1.e-3
t, W = brownian_motion(Δt, tmax);
V = sqrt(Δt / 4) * randn(length(t))
@show ito_integral(W)

@show strato_integral(W, V);

# %%
@show ito_oneliner(W)

@show strato_oneliner(W, V);

# %%
# Exact stochastic integrals for the Wiener process
ito_exact(W, tmax) = W[end]^2 / 2 - tmax / 2
strato_exact(W) = W[end]^2 / 2

# %%
percentage_error(ito_exact(W, tmax), ito_integral(W))

# %%
percentage_error(strato_exact(W), strato_integral(W, V))

# %% [markdown]
# ## HW : Calculate the analytical value of the Ito and Strato integrals
# ## HW : Find out how the percentage error changes with Δt

# %%
function ito_integral_deterministic(f, t, W)
    s = 0.0
    for i in 1:(length(t)-1)
        ti = t[i]
        s += f(ti) * (W[i+1] - W[i])
    end
    return s
end

# %%
tmax, Δt, nens = 1, 1.e-3, 10000
f(t) = t^2
t, df = brownian_motion(Δt, tmax, nens)
ift = map(Wi -> ito_integral_deterministic(f, t, Wi), eachcol(df));

# %%
fig, ax = figax()
plot_probability_distribution!(ax, ift, label = "Numerical")
plot_normal_distribution!(ax, 1.5; σ = sqrt(1 / 5), color = :black, linewidth = 2)
axislegend(ax)
fig
