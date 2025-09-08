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
includet("src/etd_factors.jl")
includet("src/repeated_vector.jl")
includet("src/sde_algorithms.jl")
includet("src/oscillator.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

# %%
# Noise implementation
shkerin_sibiryakov_noise(h) = (1/h, h/12, h^3/720) .* (randn(), randn(), randn())

function shkerin_sibiryakov_noise_convergence!(ΔW, ϵ, h, K, W)
    f1, f2 = (h/2-ϵ/2), (h^2/12 - ϵ*h/4 + ϵ^2/6)
    for i in 1:size(ΔW)[1]
        s = (i-1)*K+1
        dW1, dW2, dW3 = W[s+K, 1] - W[s, 1], W[s+K, 2] - W[s, 2], W[s+K, 3] - W[s, 3]
        ΔW[i, 1], ΔW2, ΔW3 = dW1/h, dW2 + f1 * dW1, dW3 + f1 * dW2 + f2 * dW1

        for k in 1:K
            tk, dW1, dW2 = (k-1) * ϵ, W[s+k, 1] - W[s+k-1, 1], W[s+k, 2] - W[s+k-1, 2]
            ΔW2 += - tk * dW1
            ΔW3 += - tk * dW2 + (-h*tk/2 + tk^2/2 + ϵ*tk/2) * dW1
        end
        ΔW[i, 2] = ΔW2 / h
        ΔW[i, 3] = ΔW3 / h
    end
end

# %%
function shkerin_sibiryakov!(x, t, p, fhD, frhs, dfrhs)
    h, D = fhD(t, p)
    fη = sqrt(2 * D)

    x[1] = p.x0
    W = zero(t)
    @inbounds for i in 2:length(t)
        xp = x[i-1]
        d1, d2, d3 = shkerin_sibiryakov_noise(h)
        x[i] = xp + h * (frhs(xp, p) + fη * (d1 + d2 * dfrhs(xp, p) + d3 * dfrhs(xp, p)^2))
        W[i] = W[i-1] + h * d1
    end
    return W
end

function oscillator_SS!(x, t, W, p)
    shkerin_sibiryakov!(
        x, t, p,
        (t, p) -> (t[2] - t[1], p.Γ * p.T),
        (x, p) -> -p.Γ * (x + p.b * x^3),
        (x, p) -> -p.Γ * (x + 3 * p.b * x^2)
    )
end

# %%
pars = (; x0 = 0.5, tmax = 10.0, nens = 100, T = 0.1, Γ = 1.0, b = 1.e-1, z = 3);
t, W = brownian_motion(1.e-2, pars.tmax)
x = zero(W)

fig, ax = figax(a = 2, xlabel = "t", ylabel = "x(t)")
W = oscillator_SS!(x, t, W, pars)
lines!(ax, t, x)
oscillator_EM!(x, t, W, pars)
lines!(ax, t, x)
ax.title="Comparison between EM and SS"
fig

# %%
function shkerin_sibiryakov_convergence(Δt, p, fhD, frhs, dfrhs)
    function _noise(ϵ, p)
        t, W = brownian_motion(ϵ, p.tmax, 3)
        W = Matrix(W)
        @. W[:, 2] *= sqrt(ϵ^2/12)
        @. W[:, 3] *= sqrt(ϵ^4/720)
        return t, W
    end

    @inline _f(x, p, fη, d) = @inbounds frhs(x, p) + fη * (d[1] + p.o[2] * d[2] * dfrhs(x, p) + p.o[3] * d[3] * dfrhs(x, p)^2)

    function _march(t, W, p, h)
        ϵ, D = fhD(t, p)
        fη = sqrt(2 * D)
        K = round(Int, h/ϵ)
        N = (length(t)-1) ÷ K
        ΔW = zeros(N, 3)
        shkerin_sibiryakov_noise_convergence!(ΔW, ϵ, h, K, W)
        x = p.x0
        @inline for s in 1:N
            @views d = ΔW[s, :]
            k1 = _f(x, p, fη, d)
            k2 = _f(x + h*k1/2, p, fη, d)
            k3 = _f(x + h*k2/2, p, fη, d)
            k4 = _f(x + h*k3, p, fη, d)
            x = x + h * (k1 + 2k2 + 2k3 + k4)/6
        end
        return x
    end

    es = zeros(p.nens, length(Δt))
    for n in 1:p.nens
        t, W = _noise(minimum(Δt), p)
        x = zero(Δt)
        for (i, h) in enumerate(Δt)
            x[i] = _march(t, W, p, h)
            es[n, i] = abs(x[1]-x[i])
        end
    end
    return vec(mean(es, dims = 1))
end

# %%
Random.seed!(314)
# Tweak pars.o for order.
# Only works for z = 1.0
pars = (; x0 = 0.5, tmax = 1.0, nens = 50, T = 1.0, Γ = 1.0, b = 1.e-2, o = [1, 1, 1], z = 1.0);

Δt = [1.e-5, 2.e-5, 4.e-5, 1.e-4, 2.e-4, 4.e-4, 1.e-3, 2.e-3, 4.e-3, 1.e-2, 2.e-2, 4.e-2, 1.e-1]

function oscillator_SS_convergence(Δt, p)
    shkerin_sibiryakov_convergence(
        Δt, p,
        (t, p) -> (t[2] - t[1], p.Γ * p.T),
        (x, p) -> -p.Γ * (x + p.b * x^p.z),
        (x, p) -> -p.Γ * (1 + p.z * p.b * x^(p.z-1))
    )
end
es = oscillator_SS_convergence(Δt, pars);

# %%
fig, ax = figax(xscale = log10, yscale = log10)
scatterlines!(ax, Δt[2:end], es[2:end]; markersize = 20)
lines!(ax, Δt, Δt .^ 3)
fig

# %%
fig, ax = figax(xscale = log10, yscale = log10)
scatterlines!(ax, Δt[2:end], es[2:end]; markersize = 20)
pars.o[3] = 0
@show pars
es = oscillator_SS_convergence(Δt, pars);
scatterlines!(ax, Δt[2:end], es[2:end]; markersize = 20)
pars.o[2] = 0
@show pars
es = oscillator_SS_convergence(Δt, pars);
scatterlines!(ax, Δt[2:end], es[2:end]; markersize = 20)
lines!(ax, Δt, Δt .^ 1)
fig

# %%
using Chairmarks, BenchmarkTools
Δt = [1.e-2, 2.e-2, 4.e-2, 1.e-1]
@benchmark oscillator_SS_convergence(Δt, pars)

# %%
@be oscillator_SS_convergence(Δt, pars)

# %%
# Stochastic pendulum convergence
function symplectic4factors(h)
    θ = 1/(2-2^(1/3))
    a = @. 0.5h * (θ, 1-θ, 1-θ, θ)
    b = @. h * (θ, 1-2θ, θ, 0)
    return (a, b)
end
function symplectic4(x0, v0, p, f::F) where {F}
    x, v = x0, v0
    for (ai, bi) in zip(p.a, p.b)
        x = x + ai*v
        v = v + bi*f(x, p)
    end
    return x, v
end

# Numerical integration of a simple oscillator.
h = 1.e-2
a, b = symplectic4factors(h)
p = (; k = 1.0, a, b)

x, v = [1.0], [0.0]
for i in 2:1001
    xi, vi = symplectic4(x[i-1], v[i-1], p, (x, p) -> -p.k*x)
    push!(x, xi)
    push!(v, vi)
end
E = @. 0.5*(x^2 + v^2)
println(extrema(E .- 0.5))

# %%
