# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: sohrab 1.10.10
#     language: julia
#     name: sohrab-1.10
# ---

# %%
# Imports and setup
import Pkg;
Pkg.activate(".");
using Revise, Printf, CairoMakie, DataFrames, StatsBase, Random
includet("src/plotting.jl")
includet("src/brownian.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

# %% [markdown]
# # 1. Wiener Process (Brownian Motion)

# %%
tmax, Δt, nens = 1, 1.e-2, 1000
t, df = brownian_motion(Δt, tmax, nens)
show(df; allrows = false)

# %%
fig, ax = figax(a = 2, title = "Ensemble of Brownian motion", xlabel = "t", ylabel = "W(t)")
for e in 1:10
    lines!(ax, t, df[!, e])
end
fig

# %%
fig, ax = figax(a = 2, title = "Mean and Variance", xlabel = "t")
lines!(ax, t, zero(t), color = :black)
lines!(ax, t, t, color = :black)
lines!(ax, t, mean.(eachrow(df)), label = "mean")
lines!(ax, t, var.(eachrow(df)), label = "variance")
axislegend(ax, position = :lt)
fig

# %% [markdown]
# ## HW : Try increasing nens and see how the mean and variance of W(t) change with nens.
# ## HW : Find that autocorrelation of the Wiener process , i.e. <W(t) W(s)> and compare with the analytical expression.

# %% [markdown]
# # 2. Functions of Brownian Motion

# %%
f(t, W, a, b) = @. exp(a * t + b * W)
meanf(t, a, b) = @. exp((a + b^2 / 2) * t)

fig, ax = figax(a = 2, h = 5, title = "f(t, W(t)) = exp(t + W(t)/2)", xlabel = "t", ylabel = "f(t, W(t))")

dff = DataFrame(map(W -> f(t, W, 1, 1 / 2), eachcol(df)), :auto)

for e in 1:10
    lines!(ax, t, dff[!, e], color = (colors[1], 0.2), linewidth = 1.0)
end

ax.limits = (-0.1, 1.1, 0.5, 4.0)
ax.yticks = 1:1:4
lines!(ax, t, mean.(eachrow(dff)), label = "Numerical Average")
lines!(ax, t, meanf(t, 1, 1 / 2), label = "Analytical", color = colors[2])
axislegend(ax, position = :lt)
fig

# %% [markdown]
# ## HW : Find the analytical expression for the expected value of f(t, W)
