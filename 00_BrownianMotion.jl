# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Julia 1.10.8
#     language: julia
#     name: julia-1.10
# ---

# Imports and setup
import Pkg;
Pkg.activate(".");
using Revise, Printf, CairoMakie, DataFrames, StatsBase, Random, UnPack
includet("src/plotting.jl")
includet("src/brownian.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

# # 1. Wiener Process (Brownian Motion)

tmax, Δt, nens = 1, 1.e-2, 1000
t, df = brownian_motion(Δt, tmax, nens)
show(df; allrows = false)

fig, ax = figax(a = 2, title = "Ensemble of Brownian motion", xlabel = "t", ylabel = "W(t)")
for e in 1:10
    lines!(ax, t, df[!, e])
end
fig

fig, ax = figax(a = 2, title = "Mean and Variance", xlabel = "t")
Wμ, Wσ = mean.(eachrow(df)), var.(eachrow(df))
lines!(ax, t, Wμ, label = "mean")
lines!(ax, t, Wσ, label = "variance")
lines!(ax, t, zero(t), color = :black)
lines!(ax, t, t, color = :black)
axislegend(ax, position = :lt)
fig

# ## HW : Try increasing nens and see how the mean and variance of W(t) change with nens.
# ## HW : Find that autocorrelation of the Wiener process , i.e. mean(W(t) W(s)) and compare with the analytical expression.

# # 2. Functions of Brownian Motion

# +
f(t, W) = @. exp(t + W / 2)

fig, ax = figax(a = 2, h = 5, title = "f(t, W(t)) = exp(t + W(t)/2)", xlabel = "t", ylabel = "f(t, W(t))")

dff = DataFrame(map(W -> f(t, W), eachcol(df)), :auto)

for e in 1:10
    lines!(ax, t, dff[!, e], color = (colors[1], 0.2), linewidth = 1.0)
end

ax.limits = (-0.1, 1.1, 0.5, 4.0)
ax.yticks = 1:1:4
lines!(ax, t, mean.(eachrow(dff)), label = "Numerical Average")
lines!(ax, t, exp.(9t / 8), label = "Analytical", color = colors[2])
axislegend(ax, position = :lt)
fig
# -

# ## HW : Find the analytical expression for the expected value of f(t, W)
