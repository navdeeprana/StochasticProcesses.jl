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

# # 3. Riemann integrals

# +
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
# -

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

# ## HW : Find analytical expression for the left and mid rules.
# ## HW : Implement the right Riemann integral and find out how it converges.

# # 3. Stochastic integrals

# +
# Stochastic integrals for the Wiener process W
function ito_integral(W)
    N = length(W)
    sum = 0
    for i in 1:N-1
        sum += W[i] * (W[i+1] - W[i])
    end
    return sum
end

function strato_integral(W, Δt)
    N = length(W)
    sum = 0
    for i in 1:N-1
        sum += 0.5 * (W[i+1] + W[i] + sqrt(Δt) * randn()) * (W[i+1] - W[i])
    end
    return sum
end
# -

tmax, Δt = 1, 1.e-4
t, W = brownian_motion(Δt, tmax);
@show ito_integral(W)
@show strato_integral(W, Δt);

# One line variants of the above functions
@inbounds ito_oneliner(W) = sum(W[i] * (W[i+1] - W[i]) for i in 1:length(W)-1)
@inbounds strato_oneliner(W, Δt) =
    sum((W[i+1] + W[i] + sqrt(Δt) * randn()) * (W[i+1] - W[i]) / 2 for i in 1:length(W)-1)

@show ito_oneliner(W)
@show strato_oneliner(W, t[2] - t[1]);

# Exact stochastic integrals for the Wiener process
ito_exact(W, tmax) = W[end]^2 / 2 - tmax / 2
strato_exact(W) = W[end]^2 / 2

@show ito_exact(W, tmax)
@show strato_exact(W);
