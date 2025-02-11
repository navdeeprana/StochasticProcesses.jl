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
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

function moran_process_gillespie_trajectory(pars)
    @unpack k1, k2, tmax, N, NA0 = pars
    t, Z = Float64[], Int[]
    tnow, nAnow = 0.0, NA0
    @inbounds while (tnow < tmax) && (0 <= nAnow <= N)
        push!(t, tnow)
        push!(Z, nAnow)
        fNA = nAnow * (N - nAnow) 
        propensity = (k1+k2)*fNA
        tnow = tnow - log(rand())/propensity
        rand() < k1 * fNA / propensity ? nAnow += 1 : nAnow -= 1
    end
    return t, Z
end

function moran_process_gillespie_end(pars, NA0)
    @unpack k1, k2, nens, tmax, N, tmax = pars
    t, Z = zeros(nens), zeros(Int, nens)
    for n in 1:nens
        tnow, nAnow = 0.0, NA0
        while (tnow < tmax) && (0 < nAnow < N)
            fNA = nAnow * (N - nAnow) 
            propensity = (k1+k2)*fNA
            tnow = tnow - log(rand())/propensity
            rand() < k1 * fNA / propensity ? nAnow += 1 : nAnow -= 1
        end
        t[n], Z[n] = tnow, nAnow
    end
    return t, Z
end

langevin_params(k1, k2, N) = (k1 - k2) * N, sqrt(k1 + k2)
extinction_probability(γ, μ, ρ0) = expm1(2 * γ * (1 - ρ0) / μ^2) / expm1(2 * γ / μ^2)
extinction_probability(Z) = sum(Z .== 0) ./ length(Z)

pars = (k1 = 0.101, k2 = 0.100, tmax = 5, nens = 1000, N = 10000)
@show γ, μ = langevin_params(pars.k1, pars.k2, pars.N)
extinction_probability(γ, μ, 1.e-2)

NA0 = 50:50:500
Zens = [moran_process_gillespie_end(pars, NA0i)[2] for NA0i in NA0];

pext = [extinction_probability(Z) for Z in Zens]
pext_an = [extinction_probability(γ, μ, NA0i/pars.N) for NA0i in NA0];

fig, ax = figax(xlabel="ρ0", ylabel="Pext(ρ0)")
scatter!(ax, NA0./pars.N, pext, label = "Gillespie")
lines!(ax, NA0./pars.N, pext_an, color = :black, label = "Analytical")
axislegend(ax)
fig
