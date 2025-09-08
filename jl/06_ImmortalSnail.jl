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
includet("src/etd.jl")
includet("src/sevector.jl")
includet("src/oscillator.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

# %%
function immortal_snail_step(z0, pars)
    (; a, v, r, L, Δt) = pars
    reset = false
    if abs(z0) < a
        if rand() < r * Δt
            reset = true
        end
    end
    reset ? zt = L : zt = z0 - v*Δt + randn()*sqrt(Δt)
    return zt
end

# %%
pars = (a = 1.0, v = 2.0, r = 1.0, L = 10.0, Δt = 1.e-2)
zt = [pars.L]
for i in 1:10000
    push!(zt, immortal_snail_step(zt[end], pars))
end

# %%
fig, ax = figax()
lines!(ax, zt)
fig

# %%
