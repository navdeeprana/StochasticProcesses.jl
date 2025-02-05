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

# +
function density_em(pars, t, W)
    @unpack μ = pars
    Δt = t[2] - t[1]
    ρ = zeros(length(W))
    ρ[1] = 1
    @inbounds for i in 2:length(W)
        ρp, dW = ρ[i-1], W[i] - W[i-1]
        if ρp > 0
            ρ[i] = ρp + μ * sqrt(ρp) * dW
        else
            ρ[i] = 0
        end
    end
    return ρ
end

pars = (tmax = 2.0, μ = 1.0)
t, W = brownian_motion(1.e-2, pars.tmax)
ρ = density_em(pars, t, W);

# +
fig, ax = figax(h = 5, xlabel = "t", ylabel = "ρ(t)")
hlines!(ax, 0, color = (:black, 0.5), linestyle = :dash)

lines!(ax, t, ρ)
fig
# -

function density_transformed_em(pars, t, W)
    @unpack μ = pars
    Δt = t[2] - t[1]
    u = zeros(length(W))
    u[1] = 1
    @inbounds for i in 2:length(W)
        up, dW = u[i-1], W[i] - W[i-1]
        if up > 1.e-3
            u[i] = up - Δt * μ^2 / (8 * up) + (μ / 2) * dW
        else
            u[i] = 0
        end
    end
    return u .^ 2
end

# +
fig, ax = figax(h = 7, xlabel = "t", ylabel = "ρ(t)")
hlines!(ax, 0, color = (:black, 0.5), linestyle = :dash)
ρtrans = density_transformed_em(pars, t, W)

lines!(ax, t, ρ)
lines!(ax, t, ρtrans)
fig

# +
using Distributions: Gamma, Poisson

function noise_dornic!(ρ, λ)
    @inline randgamma(p) = p > 0.0 ? rand(Gamma(p)) : 0.0
    return randgamma(rand(Poisson(λ * ρ))) / λ
end

function density_dornic_exact(pars, Δt)
    @unpack tmax, μ = pars
    λ = 2 / (μ^2 * Δt)
    N = round(Int, tmax / Δt)
    t = Δt .* (0:1:N)
    ρ = zeros(N + 1)
    ρ[1] = 1
    @inbounds for i in 2:N+1
        ρ[i] = noise_dornic!(ρ[i-1], λ)
    end
    return t, ρ
end
# -

fig, ax = figax(h = 7, xlabel = "t", ylabel = "ρ(t)")
hlines!(ax, 0, color = (:black, 0.5), linestyle = :dash)
t, ρ = density_dornic_exact(pars, 1.e-3)
lines!(ax, t, ρ)
fig

# +
function density_transformed_em_end(pars, Δt)
    @unpack μ, tmax, nens = pars
    N = round(Int, tmax / Δt)
    u = zeros(nens)
    for n in 1:nens
        un = 1.0
        for i in 2:N+1
            if abs(un) > 1.e-3
                un = un - Δt * μ^2 / (8 * un) + (μ / 2) * sqrt(Δt) * randn()
            else
                un = 0
                continue
            end
        end
        u[n] = un
    end
    return u .^ 2
end

function density_dornic_exact_end(pars, Δt)
    @unpack μ, tmax, nens = pars
    N = round(Int, tmax / Δt)
    λ = 2 / (μ^2 * Δt)
    ρ = zeros(nens)
    for n in 1:nens
        ρn = 1.0
        for i in 1:N
            ρn = noise_dornic!(ρn, λ)
        end
        ρ[n] = ρn
    end
    return ρ
end
# -

pars = (tmax = 1.0, nens = 1, μ = 1.0)
ρ1 = density_dornic_exact_end(pars, 2.e-1);
# ρ2 = density_transformed_em_end(pars, 1.e-3);

# +
using SpecialFunctions, LinearAlgebra

plot_probability_distribution!(ax, ρ, bins; kw...) = stephist!(ax, ρ; normalization = :pdf, bins, kw...)

P_analytical(ρ, λ; ρ0 = 1) = λ * exp(-λ * (ρ0 + ρ)) * sqrt(ρ0 / ρ) * besseli(1, 2 * λ * sqrt(ρ0 * ρ))

function partial_probability(x0, x1, λ; N = 500)
    x = LinRange(x0, x1, N)
    P = @. P_analytical(x, λ)
    return sum(P) * (x[2] - x[1])
end

function plot_analytical_distribution!(ax, pars, ρ; kw...)
    λ = 2 / (pars.μ^2 * pars.tmax)
    Δρ = ρ[2] - ρ[1]
    P = @. P_analytical(ρ, λ)
    # println(partial_probability(Δρ/2, Δρ, λ) / Δρ)
    P[1] = (1 - Δρ * sum(P[2:end])) / Δρ
    @show abs(P[1] - exp(-λ) / Δρ)/P[1]
    lines!(ax, ρ, P; label = "Analytical", kw...)
    scatter!(ax, 0, P[1])
    scatter!(ax, 0, exp(-λ)/ Δρ)
    return P
end
# -

pars = (tmax = 1.0, nens = 20000, μ = 1.0)
ρ1 = density_dornic_exact_end(pars, 1.e-1);
ρ2 = density_transformed_em_end(pars, 1.e-3);

# +
fig, ax = figax()
bins = LinRange(0, maximum(ρ1), 1000)
plot_probability_distribution!(ax, ρ1, bins)
# plot_probability_distribution!(ax, ρ2, bins)

plot_analytical_distribution!(ax, pars, bins)
ax.limits = (-0.1, 0.5, nothing, nothing)
axislegend(ax)
fig
# -

pars = (tmax = 1.0, nens = 100000, μ = 1.0)
ρ1 = density_dornic_exact_end(pars, 1.e-1);
ρ2 = density_transformed_em_end(pars, 1.e-3);

fig, ax = figax(h = 10)
bins = LinRange(0, maximum(ρ1), 2000)
plot_probability_distribution!(ax, ρ1, bins)
plot_probability_distribution!(ax, ρ2, bins)
plot_analytical_distribution!(ax, pars, bins)
ax.limits = (-0.1, 2.0, nothing, nothing)
fig

# +
# using StatsBase
# h = fit(Histogram, ρ1, bins; closed=:left)
# h = normalize(h)
# lines!(ax, h.edges[1][1:end-1], h.weights)
# -


