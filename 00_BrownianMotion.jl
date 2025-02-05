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
    N, s = length(W), 0.0
    for i in 1:N-1
        s += W[i] * (W[i+1] - W[i])
    end
    return s
end

function strato_integral(W, V)
    N, s = length(W), 0.0
    for i in 1:N-1
        s += (0.5*(W[i+1] + W[i]) + V[i]) * (W[i+1] - W[i])
    end
    return s
end

# One line variants of the above functions
@inbounds ito_oneliner(W) = sum(W[i] * (W[i+1] - W[i]) for i in 1:length(W)-1)

@inbounds strato_oneliner(W, V) = sum((0.5*(W[i+1] + W[i]) + V[i]) * (W[i+1] - W[i]) for i in 1:length(W)-1)

percentage_error(a,b) = 100 * abs((a-b)/a)
# -

# ## HW : Prove the expression for $W((t_j + t_{j+1})/2)$

# +
tmax, Δt = 1, 1.e-3
t, W = brownian_motion(Δt, tmax);
V = sqrt(Δt/4) * randn(length(t))
@show ito_integral(W)

@show strato_integral(W, V);

# +
@show ito_oneliner(W)

@show strato_oneliner(W, V);
# -

# Exact stochastic integrals for the Wiener process
ito_exact(W, tmax) = W[end]^2 / 2 - tmax / 2
strato_exact(W) = W[end]^2 / 2

percentage_error(ito_exact(W, tmax), ito_integral(W))

percentage_error(strato_exact(W), strato_integral(W, V))

# ## HW : Calculate the analytical value of the Ito and Strato integrals
# ## HW : Find out how the percentage error changes with Δt

function ito_integral_deterministic(f, t, W)
    s = 0.0
    for i in 1:length(t)-1
        ti = t[i]
        s += f(ti) * (W[i+1] - W[i])
    end
    return s
end

tmax, Δt, nens = 1, 1.e-3, 10000
f(t) = t^2
t, df = brownian_motion(Δt, tmax, nens)
ift = map(Wi -> ito_integral_deterministic(f, t, Wi), eachcol(df));

fig, ax = figax()
plot_probability_distribution!(ax, ift, label="Numerical")
plot_normal_distribution!(ax, 1.5; σ = sqrt(1/5), color=:black, linewidth=2)
axislegend(ax)
fig
