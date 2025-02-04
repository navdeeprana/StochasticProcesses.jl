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
function etd_factors(Δt, Γ, b, T)
    c, D = -Γ, Γ * T
    f = (exp(Δt * c), -Γ * b * expm1(Δt * c) / c, sqrt((D / c) * (exp(2 * c * Δt) - 1)))
end

function overdamped_oscillator_etd(pars, Δt)
    @unpack nens, tmax, Γ, b, T = pars
    iters = round(Int, tmax / Δt)
    f = etd_factors(Δt, Γ, b, T)
    x, η = zeros(nens), zeros(nens)
    for t in 1:iters
        randn!(η)
        @. x = f[1] * x + f[2] * x^3 + f[3] * η
    end
    return x
end

function overdamped_oscillator_euler(pars, Δt)
    @unpack nens, tmax, Γ, b, T = pars
    iters = round(Int, tmax / Δt)
    fη = sqrt(2 * Γ * T * Δt)
    x, η = zeros(nens), zeros(nens)
    for t in 1:iters
        randn!(η)
        @. x = x + Δt * (-Γ * (x + b * x^3)) + fη * η
    end
    return x
end

plot_probability_distribution!(ax, x_eq; bins = 100, kw...) = stephist!(ax, x_eq; normalization = :pdf, bins, kw...)

function plot_boltzmann_distribution!(ax, pars, xm; kw...)
    x = LinRange(-xm, xm, 1000)
    P = @. exp(-(x^2 / 2 + pars.b * x^4 / 4) / pars.T)
    P = P / sum(P * (x[2] - x[1]))
    lines!(ax, x, P; label = "Boltzmann", kw...)
end
# -

pars = (; tmax = 10, nens = 10000, T = 6.0, Γ = 5.0, b = 1.e-2);

fig, axes = figax(nx = 2, h = 5, xlabel = "x", ylabel = "P(x)")
for (ax, Δt) in zip(axes, [1.e-2, 1.e-1])
    x1 = overdamped_oscillator_etd(pars, Δt)
    x2 = overdamped_oscillator_euler(pars, Δt)
    plot_boltzmann_distribution!(ax, pars, maximum(x1); color = :black)
    plot_probability_distribution!(ax, x1; linewidth = 2, label = "Stochastic ETD")
    plot_probability_distribution!(ax, x2; linewidth = 2, label = "Euler-Murayama")
    ax.limits = (-9, 9, nothing, nothing)
    ax.title = @sprintf "Δt = %.2f" Δt
end
axislegend(axes[2], position = :cb)
fig

function oscillator_etd_final(t, W, pars)
    Δt = t[2] - t[1]
    f = etd_factors(Δt, pars.Γ, pars.b, pars.T)
    x = 0
    @inbounds for i in 2:length(W)
        x = f[1] * x + f[2] * x^3 + f[3] * (W[i] - W[i-1])
    end
    return x
end

function setd_convergence(pars; scale = 4)
    @unpack tmax, nens = pars
    Δt = @. 1 / 2^(2:8)

    N = @. round(Int, tmax / Δt)
    t, W = brownian_motion(Δt[end] / scale, tmax, nens)

    XanT = map(Wi -> oscillator_etd_final(t, Wi, pars), eachcol(W))

    estrong, eweak = Float64[], Float64[]
    for Ni in N
        skip = length(t) ÷ Ni
        @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
        XT = map(Wni -> oscillator_etd_final(tn, Wni, pars), eachcol(Wn))
        push!(estrong, mean(@. abs(XanT .- XT)))
        push!(eweak, abs(mean(XanT) - mean(XT)))
    end
    return N, estrong, eweak
end

fig, ax = figax(h = 5, xscale = log2, yscale = log2)
pars = (; tmax = 1, nens = 20000, T = 6.0, Γ = 5.0, b = 1.e-2);
N, es, ew = setd_convergence(pars; scale = 2)
plot_convergence(fig, ax, N, es, ew, -0.5, -1.0)
fig


