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
#     display_name: Julia 1.10.8
#     language: julia
#     name: julia-1.10
# ---

# %%
# Imports and setup
import Pkg;
Pkg.activate(".");
using Revise, Printf, CairoMakie, DataFrames, StatsBase, Random
includet("src/plotting.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

# %% [markdown]
# # Dornic algorithm for Moran process

# %%
using Distributions: Gamma, Poisson

function noise_dornic!(ρ, λ)
    @inline randgamma(p) = p > 0.0 ? rand(Gamma(p)) : 0.0
    return randgamma(rand(Poisson(λ * ρ))) / λ
end

# %%
f(x) = sqrt(x * (1 - x))
fapp(x) = x < 0.5 ? sqrt(x) : sqrt(1 - x)

x = 0:0.001:1
fig, ax = figax(xlabel = "ρ", title = "Approximating noise")
lines!(ax, x, f.(x), label = "Exact")
lines!(ax, x, fapp.(x), label = "Approximation")
axislegend(ax, position = :cb)
fig

# %%
function moran_process_end(pars, Δt, ρ0, fapprox)
    (; γ, μ, tmax, nens) = pars
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
    (; γ, μ, tmax, nens) = pars
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

# %%
# Analytical expression for the extinction probability
extinction_probability(γ, μ, ρ0) = expm1(2 * γ * (1 - ρ0) / μ^2) / expm1(2 * γ / μ^2)

# Numerical method for an ensemble
extinction_probability(x) = sum(x .< 1.e-10) ./ length(x)

# %%
pars = (γ = 1.0, μ = 1.0, tmax = 2.0, nens = 50000)
ρ0 = 0.05:0.05:0.50
ρens1 = [moran_process_end(pars, 1.e-2, ρ0i, fapprox1) for ρ0i in ρ0]
ρens2 = [moran_process_end(pars, 1.e-2, ρ0i, fapprox2) for ρ0i in ρ0];

# %%
pext1 = [extinction_probability(ρ) for ρ in ρens1]
pext2 = [extinction_probability(ρ) for ρ in ρens2]
pext_an = extinction_probability.(pars.γ, pars.μ, ρ0)

fig, ax = figax(xlabel = "ρ0", ylabel = "Pext(ρ0)")
scatter!(ax, ρ0, pext1, label = "Approx")
scatter!(ax, ρ0, pext2, label = "Half-approx")
lines!(ax, ρ0, pext_an, color = :black, label = "Analytical")
axislegend(ax)
fig

# %%
pars = (γ = 1.0, μ = 1.0, tmax = 2.0, nens = 10)
ρens1 = moran_process_trajectory(pars, 1.e-2, 0.2, fapprox1);
ρens2 = moran_process_trajectory(pars, 1.e-2, 0.2, fapprox2);

# %%
fig, ax = figax(nx = 2, h = 5, a = 2, xlabel = "t", ylabel = "ρ(t)")
ax[1].title = "Large reversals for approx"
ax[2].title = "Deterministic for half-approx"
for n in 1:pars.nens
    lines!(ax[1], ρens1[n, :])
    lines!(ax[2], ρens2[n, :])
end
fig

# %%
pars = (γ = 1.0, μ = 0.1, tmax = 5.0, nens = 20)
ρens1 = moran_process_trajectory(pars, 1.e-2, 0.2, fapprox1);
ρens2 = moran_process_trajectory(pars, 1.e-2, 0.2, fapprox2);

# %%
fig, ax = figax(nx = 2, h = 5, a = 1.8, xlabel = "t", ylabel = "ρ(t)")
fig[0, :] = Label(fig, "Works better for large γ/μ^2")
ax[1].title = "approx"
ax[2].title = "half approx"
for n in 1:pars.nens
    lines!(ax[1], ρens1[n, :])
    lines!(ax[2], ρens2[n, :])
end
fig

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## HW : Implement a better deterministic algorithm for the SODE of Moran process
# ## HW : Implement the general distribution with source terms in Dornic et. al, PRL
# ## HW : Implement the Voter model as discussed in Dornic et. al, PRL

# %% [markdown]
# # Gillespie algorithm for Moran process

# %%
function moran_process_gillespie_trajectory(pars, NA0)
    (; k1, k2, tmax, N) = pars
    t, Z = Float64[], Int[]
    tnow, nAnow = 0.0, NA0
    @inbounds while (tnow < tmax) && (0 <= nAnow <= N)
        push!(t, tnow)
        push!(Z, nAnow)
        fNA = nAnow * (N - nAnow)
        propensity = (k1 + k2) * fNA
        tnow = tnow - log(rand()) / propensity
        rand() < k1 * fNA / propensity ? nAnow += 1 : nAnow -= 1
    end
    return t, Z
end

function moran_process_gillespie_end(pars, NA0)
    (; k1, k2, nens, tmax, N, tmax) = pars
    t, Z = zeros(nens), zeros(Int, nens)
    for n in 1:nens
        tnow, nAnow = 0.0, NA0
        while (tnow < tmax) && (0 < nAnow < N)
            fNA = nAnow * (N - nAnow)
            propensity = (k1 + k2) * fNA
            tnow = tnow - log(rand()) / propensity
            rand() < k1 * fNA / propensity ? nAnow += 1 : nAnow -= 1
        end
        t[n], Z[n] = tnow, nAnow
    end
    return t, Z
end

# %%
pars = (k1 = 0.11, k2 = 0.10, tmax = 1, nens = 10, N = 10000)
fig, ax = figax(a = 2, xlabel = "t", ylabel = "ρ(t)")
for n in 1:pars.nens
    t, Z = moran_process_gillespie_trajectory(pars, 100)
    lines!(ax, t, Z ./ pars.N)
end
fig

# %% [markdown]
# ## HW: Implement an averaging procedure for the trajectories obtained from Gillespie algorithm

# %%
# Given microscopic rates, compute the parameters of the Langevin equation
langevin_params(k1, k2, N) = (k1 - k2) * N, sqrt(k1 + k2)

# %%
pars = (k1 = 0.101, k2 = 0.100, tmax = 5, nens = 1000, N = 10000)
@show γ, μ = langevin_params(pars.k1, pars.k2, pars.N)
extinction_probability(γ, μ, 1.e-2)

# %%
using ProgressMeter
NA0 = 50:50:500
Zens = @showprogress [moran_process_gillespie_end(pars, NA0i)[2] for NA0i in NA0];

# %%
pext = [extinction_probability(Z) for Z in Zens]
pext_an = [extinction_probability(γ, μ, NA0i / pars.N) for NA0i in NA0];

# %%
fig, ax = figax(xlabel = "ρ0", ylabel = "Pext(ρ0)", title = "Extinction probability for the Moran process")
scatter!(ax, NA0 ./ pars.N, pext, label = "Gillespie")
lines!(ax, NA0 ./ pars.N, pext_an, color = :black, label = "Analytical")
axislegend(ax)
fig

# %% [markdown]
# ## HW: Implement Gillespie Tau algorithm
