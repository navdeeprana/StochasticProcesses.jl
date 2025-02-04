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
includet("src/integrals.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

# # 1. Brownian Motion

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

# # 3. Stochastic Integrals

f(x) = 3x^2
@show riemann_integral_left(f, 0, 1, 20);
@show riemann_integral_mid(f, 0, 1, 20);

# +
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
# -

tmax, Δt = 1, 1.e-4
t, W = brownian_motion(Δt, tmax);
@show ito_integral(W)
@show strato_integral(W, Δt);

@show ito_oneliner(W)
@show strato_oneliner(W, t[2] - t[1]);

ito_exact(W, tmax), strato_exact(W)

tmax, Δt, nens = 1, 1.e-2, 2000
t, df = brownian_motion(Δt, tmax, nens)
t = t[1:end-1]
df2 = DataFrame(map(W -> (@. W[2:end] * W[1:end-1]), eachcol(df)), :auto)
fig, ax = figax(a = 2, h = 5)
lines!(ax, t, mean.(eachrow(df2)), label = "Numerical Average")
lines!(ax, t, t)
axislegend(ax, position = :lt)
fig


