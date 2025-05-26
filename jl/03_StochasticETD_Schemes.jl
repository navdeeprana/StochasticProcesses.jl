# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Julia 1.10.9
#     language: julia
#     name: julia-1.10
# ---

# %%
# Imports and setup
import Pkg;
Pkg.activate(".");
using Revise, Printf, CairoMakie, DataFrames, StatsBase, Random, UnPack
includet("src/plotting.jl")
includet("src/brownian.jl")
includet("src/etd.jl")
includet("src/sevector.jl")
includet("src/oscillator.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
Random.seed!(42);

# %%
pars = (; tmax = 10, nens = 10000, T = 6.0, Γ = 5.0, b = 1.e-2);

fig, axes = figax(nx = 2, h = 5, xlabel = "x", ylabel = "P(x)")
for (ax, Δt) in zip(axes, [1.e-2, 1.e-1])
    x1 = oscillator_etd1_ensemble(pars, Δt)
    x2 = oscillator_euler_ensemble(pars, Δt)
    plot_boltzmann_distribution!(ax, pars, maximum(x1); color = :black)
    plot_probability_distribution!(ax, x1; linewidth = 2, label = "Stochastic ETD")
    plot_probability_distribution!(ax, x2; linewidth = 2, label = "Euler-Murayama")
    ax.limits = (-9, 9, nothing, nothing)
    ax.title = @sprintf "Δt = %.2f" Δt
end
axislegend(axes[2], position = :cb)
fig

# %%
using OrdinaryDiffEq, StochasticDiffEq

f(u, p, t) = -p.Γ * (u + p.b * u^3)
g(u, p, t) = sqrt(2 * p.Γ * p.T)

prob = SDEProblem(f, g, 0.0, (0.0, pars.tmax), pars)
sol = solve(EnsembleProblem(prob), SRIW1(), saveat = pars.tmax, trajectories = pars.nens);

# %%
fig, ax = figax(h = 5, xlabel = "x", ylabel = "P(x)")
x = [ui.u[end] for ui in sol.u]
plot_boltzmann_distribution!(ax, pars, maximum(x); color = :black)
plot_probability_distribution!(ax, x; linewidth = 2, label = "SRIW1")
ax.limits = (-9, 9, nothing, nothing)
axislegend(ax, position = :cb)
fig

# %% [markdown]
# ## Convergence for the Anharmonic Oscillator

# %%
function noise_from_sol(sol)
    t, W = sol.u[1].W.t, DataFrame()
    for e in 1:length(sol.u)
        W[!, "W$e"] = sol.u[e].W.W
    end
    return t, W
end

function convergence_dejl(Δt, sol, algorithm::F) where {F}
    N = @. round(Int, prob.p.tmax / Δt)
    if prob.p.b == 0.0
        XanT = [final_solution(OU_analytical!, ui.W.t, ui.W.W, prob.p) for ui in sol.u]
    else
        XanT = [ui.u[end] for ui in sol.u]
    end
    mXanT = mean(XanT)

    t, W = noise_from_sol(sol)

    es, ew = Float64[], Float64[]
    for Ni in N
        skip = (length(t) - 1) ÷ Ni
        @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
        XT = map(Wni -> final_solution(algorithm, tn, Wni, pars), eachcol(Wn))
        push!(es, mean(@. abs(XanT - XT)))
        push!(ew, abs(mXanT - mean(XT)))
    end
    return (; Δt, N, es, ew)
end

# %%
Random.seed!(314)
pars = (; x0 = 0.5, tmax = 10.0, nens = 1000, T = 0.1, Γ = 1.0, b = 1.e-1);

# Δt = @. 1 / 2^(8:-1:4)
Δt = @. 1 / [10, 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000]

prob = SDEProblem(f, g, pars.x0, (0.0, pars.tmax), pars);
sol = solve(prob, EM(), adaptive = false, dt = 1.e-1, save_everystep = true, save_noise = true);

# %%
eprob = EnsembleProblem(prob)
sol = solve(
    eprob,
    EM(),
    adaptive = false,
    dt = minimum(Δt) / 10,
    save_everystep = true,
    save_noise = true,
    trajectories = pars.nens
);
@time cvg1 = convergence_dejl(Δt, sol, oscillator_EM!);
@time cvg2 = convergence_dejl(Δt, sol, oscillator_srk2!);
@time cvg3 = convergence_dejl(Δt, sol, oscillator_setd1!)
@time cvg4 = convergence_dejl(Δt, sol, oscillator_etd2rk!);

# %%
fig, ax = figax(nx = 2, h = 5, xscale = log10, yscale = log10)
for (cvg, label) in zip([cvg1, cvg2, cvg3, cvg4], ["EULER", "SRK2", "ETD", "ETD2RK"])
    if label in ["SRK2", "ETD2RK"]
        continue
    end
    @unpack Δt, es, ew = cvg
    scatterlines!(ax[1], Δt, (@. es / Δt^0); label, markersize = 20)
    scatterlines!(ax[2], Δt, (@. ew / Δt^0); label, markersize = 20)
    lines!(ax[1], Δt, @. 1.e+0 * Δt)
    lines!(ax[2], Δt, @. 1.e+1 * Δt)
end
# axislegend(position = :rb)
fig

# %% [markdown]
# ## Convergence for the linear SDE

# %%
function linearsde_analytical!(x, t, W, pars)
    @unpack α, β, T = pars
    x[1] = pars.x0
    ito = 0.0
    for i in 2:length(W)
        ito = ito + exp(α * t[i]) * (W[i] - W[i-1])
        x[i] = exp(-α * t[i]) * (pars.x0 - (β/α) * (1 - exp(α * t[i])) + sqrt(2 * T) * ito)
    end
    nothing
end
function convergence_linear(Δt, sol, algorithm::F) where {F}
    N = @. round(Int, prob.p.tmax / Δt)
    XanT = [final_solution(linearsde_analytical!, ui.W.t, ui.W.W, prob.p) for ui in sol.u]
    mXanT = mean(XanT)

    t, W = noise_from_sol(sol)

    es, ew = Float64[], Float64[]
    for Ni in N
        skip = (length(t) - 1) ÷ Ni
        @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
        XT = map(Wni -> final_solution(algorithm, tn, Wni, pars), eachcol(Wn))
        push!(es, mean(@. abs(XanT - XT)))
        push!(ew, abs(mXanT - mean(XT)))
    end
    return (; Δt, N, es, ew)
end

function convergence_linearsde_dejl(Δt, prob, algorithm)
    N = @. round(Int, prob.p.tmax / Δt)

    eprob = EnsembleProblem(prob)
    es, ew = Float64[], Float64[]
    for Δti in Δt
        sol = solve(
            eprob,
            algorithm,
            adaptive = false,
            dt = Δti,
            save_everystep = true,
            save_noise = true,
            trajectories = prob.p.nens
        )
        XT = [ui.u[end] for ui in sol.u]
        XanT = [final_solution(linearsde_analytical!, ui.W.t, ui.W.W, prob.p) for ui in sol.u]
        push!(es, mean(@. abs(XanT .- XT)))
        push!(ew, abs(mean(XanT) - mean(XT)))
    end
    return (; Δt, N, es, ew)
end

# %%
pars = (; x0 = 0.5, tmax = 2.0, nens = 1000, T = 1.0, α = 4.0, β = 0.5);
f_lin(u, p, t) = -p.α * u + p.β
g_lin(u, p, t) = sqrt(2 * p.T)

Δt = @. 1 / 2^(8:-1:4)

prob = SDEProblem(f_lin, g_lin, pars.x0, (0.0, pars.tmax), pars)
sol = solve(prob, EM(), adaptive = false, dt = 5.e-2, save_everystep = true, save_noise = true);

# eprob = EnsembleProblem(prob)
# sol = solve(eprob, EM(), adaptive = false, dt = minimum(Δt) / 8, save_everystep = true, save_noise = true, trajectories = pars.nens);

# %%
t, W = sol.t, sol.W.W

fig, ax = figax(a = 3)
lines!(ax, t, sol.u)

x = zero(W)
linearsde_EM!(x, t, W, pars)
lines!(ax, t, x)

linearsde_analytical!(x, t, W, pars)
lines!(ax, t, x; color = :black)
fig

# %%
fig, ax = figax(nx = 2, h = 5, xscale = log2, yscale = log2)

@time cvg = convergence_linearsde_dejl(Δt, prob, EM());
@unpack Δt, es, ew = cvg
scatterlines!(ax[1], Δt, es; markersize = 20)
scatterlines!(ax[2], Δt, ew; markersize = 20)

@time cvg = convergence_linearsde_dejl(Δt, prob, SRA3());
@unpack Δt, es, ew = cvg
scatterlines!(ax[1], Δt, es; markersize = 20)
scatterlines!(ax[2], Δt, ew; markersize = 20)

lines!(ax[1], Δt, @. 1.e+0 * Δt^1.0)
lines!(ax[2], Δt, @. 1.e+0 * Δt^1.0)
# axislegend(position = :rb)
fig

# %%
fig, ax = figax(nx = 2, h = 5, xscale = log2, yscale = log2)

@time cvg = convergence_linear(Δt, sol, linearsde_EM!);
@unpack Δt, es, ew = cvg
scatterlines!(ax[1], Δt, es; markersize = 20)
scatterlines!(ax[2], Δt, ew; markersize = 20)

@time cvg = convergence_linear(Δt, sol, linearsde_setd1!);
@unpack Δt, es, ew = cvg
scatterlines!(ax[1], Δt, es; markersize = 20)
scatterlines!(ax[2], Δt, ew; markersize = 20)

lines!(ax[1], Δt, @. 1.e+0 * Δt^1.0)
lines!(ax[2], Δt, @. 1.e+0 * Δt^1.0)
# axislegend(position = :rb)
fig

# %%
function noise_increment_from_sol(sol, N)
    t, W = sol.u[1].W.t[1:N:end], DataFrame()
    for e in 1:length(sol.u)
        Wi = sol.u[e].W.W[1:N:end]
        dW = Wi[2:end] .- Wi[1:(end-1)]
        W[!, "W$e"] = dW
    end
    return t, W
end
fig, ax = figax()
t, dW = noise_increment_from_sol(sol, 8);
plot_probability_distribution!(ax, vec(Matrix(dW)))
plot_normal_distribution!(ax, 0.1; μ = 0.0, σ = sqrt(t[2] - t[1]), color = :black)
fig
