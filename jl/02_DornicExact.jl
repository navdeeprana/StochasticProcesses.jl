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

# +
pars = (tmax = 3.0, μ = 1.0)
t, W = brownian_motion(1.e-2, pars.tmax)
ρ = density_em(pars, t, W);

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
fig, ax = figax(h = 5, xlabel = "t", ylabel = "ρ(t)")
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
for i in 1:5
    t, ρ = density_dornic_exact(pars, 1.e-2)
    lines!(ax, t, ρ)
end
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

# +
using SpecialFunctions, LinearAlgebra, Integrals

P_analytical(ρ, λ; ρ0 = 1) = λ * exp(-λ * (ρ0 + ρ)) * sqrt(ρ0 / ρ) * besseli(1, 2 * λ * sqrt(ρ0 * ρ))

probability_quad(ρs, ρf, λ) = solve(IntegralProblem(P_analytical, (ρs, ρf), λ), QuadGKJL()).u

probability_rsum(ρ, λ) = (ρ[2] - ρ[1]) * sum(@. P_analytical(ρ, λ))

function plot_analytical_distribution!(ax, pars, ρ; kw...)
    λ, Δρ = 2 / (pars.μ^2 * pars.tmax), ρ[2] - ρ[1]
    P = @. P_analytical(ρ, λ)
    P[1] = (1 - probability_rsum(ρ[2:end], λ)) / Δρ
    lines!(ax, ρ, P; label = "Analytical", kw...)
    scatter!(ax, 0, P[1]; kw...)
    P1_better = (1 - probability_quad(Δρ, ρ[end], λ)) / Δρ
    scatter!(ax, Δρ, P1_better; color = :red)
    return P
end
# -

pars = (tmax = 0.5, nens = 100000, μ = 0.5)
ρ1 = density_dornic_exact_end(pars, 1.e-1);
ρ2 = density_transformed_em_end(pars, 1.e-3);

# +
fig, ax = figax(xlabel = "ρ", ylabel = "P(ρ)")
bins = LinRange(0, maximum(ρ1), 500)
plot_probability_distribution!(ax, ρ1; bins, label = "Exact")
plot_probability_distribution!(ax, ρ2; bins, label = "Transformed")

plot_analytical_distribution!(ax, pars, bins, color = :black)
ax.limits = (-0.1, 2.0, nothing, nothing)
axislegend(ax)
ax.title = "tmax = 0.5, μ = 0.5"
fig
# -

pars = (tmax = 0.5, nens = 200000, μ = 1.0)
ρ1 = density_dornic_exact_end(pars, 1.e-1);
ρ2 = density_transformed_em_end(pars, 1.e-3);

fig, axis = figax(h = 5, nx = 2, xlabel = "ρ", ylabel = "P(ρ)")
bins = LinRange(0, maximum(ρ1), 500)
for ax in axis
    plot_probability_distribution!(ax, ρ1; bins, label = "Exact", linewidth = 3)
    plot_probability_distribution!(ax, ρ2; bins, label = "Transformed", linewidth = 3)
    plot_analytical_distribution!(ax, pars, bins, color = :black)
    ax.title = "tmax = 0.5, μ = 1.0"
end
axis[2].limits = (-0.02, 0.1, nothing, nothing)
axislegend.(axis)
fig

# # Moran process

# +
f(x) = sqrt(x * (1 - x))
fapp(x) = x < 0.5 ? sqrt(x) : sqrt(1 - x)

x = 0:0.001:1
fig, ax = figax(xlabel="ρ", title="Approximating noise")
lines!(ax, x, f.(x), label="Exact")
lines!(ax, x, fapp.(x), label="Approximation")
axislegend(ax, position=:cb)
fig

# +
function moran_process_end(pars, Δt, ρ0, fapprox)
    @unpack γ, μ, tmax, nens = pars
    N = round(Int, tmax / Δt)
    λ = 2 / (μ^2 * Δt)
    ρ = zeros(nens)
    for n in 1:nens
        ρn = ρ0
        for i in 1:N
            ρs = fapprox(ρn, λ)
            ρn = ρs + γ * Δt * ρs * (1 - ρs^2)
        end
        ρ[n] = ρn
    end
    return ρ
end

function moran_process_trajectory(pars, Δt, ρ0, fapprox)
    @unpack γ, μ, tmax, nens = pars
    N = round(Int, tmax / Δt)
    λ = 2 / (μ^2 * Δt)
    ρ = zeros(nens, N)
    ρ[:, 1] .= ρ0
    @inbounds for i in 2:N
        for n in 1:nens
            ρs = fapprox(ρ[n, i-1], λ)
            ρ[n, i] = ρs + γ * Δt * ρs * (1 - ρs^2)
        end
    end
    return ρ
end

fapprox1(ρn, λ) = ρn < 0.5 ? noise_dornic!(ρn, λ) : 1.0 - noise_dornic!(1.0 - ρn, λ)
fapprox2(ρn, λ) = ρn < 0.5 ? noise_dornic!(ρn, λ) : ρn

extinction_probability(γ, μ, ρ0) = expm1(2 * γ * (1 - ρ0) / μ^2) / expm1(2 * γ / μ^2)

extinction_probability(ρ) = 1 - mean(ρ)
# -

pars = (γ = 1.0, μ = 1.0, tmax = 2.0, nens = 50000)
ρ0 = 0.05:0.05:0.50
ρens1 = [moran_process_end(pars, 1.e-2, ρ0i, fapprox1) for ρ0i in ρ0]
ρens2 = [moran_process_end(pars, 1.e-2, ρ0i, fapprox2) for ρ0i in ρ0];

# +
pext1 = [extinction_probability(ρ) for ρ in ρens1]
pext2 = [extinction_probability(ρ) for ρ in ρens2]
pext_an = extinction_probability.(pars.γ, pars.μ, ρ0)

fig, ax = figax(xlabel="ρ0", ylabel="Pext(ρ0)")
scatter!(ax, ρ0, pext1, label = "Approx")
scatter!(ax, ρ0, pext2, label = "Half-approx")
lines!(ax, ρ0, pext_an, color = :black, label = "Analytical")
axislegend(ax)
fig
# -

pars = (γ = 1.0, μ = 1.0, tmax = 2.0, nens = 10)
ρens1 = moran_process_trajectory(pars, 1.e-2, 0.2, fapprox1);
ρens2 = moran_process_trajectory(pars, 1.e-2, 0.2, fapprox2);

fig, ax = figax(nx=2,h=5, a = 2, xlabel="t", ylabel="ρ(t)")
ax[1].title = "Large reversals for approx"
ax[2].title = "Deterministic for half-approx"
for n in 1:pars.nens
    lines!(ax[1], ρens1[n, :])
    lines!(ax[2], ρens2[n, :])
end
fig

pars = (γ = 1.0, μ = 0.1, tmax = 5.0, nens = 20)
ρens1 = moran_process_trajectory(pars, 1.e-2, 0.2, fapprox1);
ρens2 = moran_process_trajectory(pars, 1.e-2, 0.2, fapprox2);

fig, ax = figax(nx=2,h=5,a = 1.8, xlabel="t", ylabel="ρ(t)")
fig[0, :] = Label(fig, "Works better for small μ")
ax[1].title = "approx"
ax[2].title = "half approx"
for n in 1:pars.nens
    lines!(ax[1], ρens1[n, :])
    lines!(ax[2], ρens2[n, :])
end
fig

# ## HW : Implement a better deterministic algorithm for the SODE of Moran process
# ## HW : Implement the Voter model as discussed in Dornic et. al, PRL
