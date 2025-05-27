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
includet("src/convergence.jl")
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
Random.seed!(314)
pars = (; x0 = 1.0, tmax = 5.0, nens = 100, T = 5.e-2, Γ = 1.0, b = 1.e-1);

Δt = @. 1 / 2^(11:-1:2)
# Δt = @. 1 / [10, 20, 40, 50, 80, 100, 200, 400, 500, 800, 1000, 2000]

prob = SDEProblem(f, g, pars.x0, (0.0, pars.tmax), pars);
sol = solve(prob, EM(), adaptive = false, dt = 1.e-2, save_everystep = true, save_noise = true);

# %%
eprob = EnsembleProblem(prob)
sol = solve(
    eprob,
    EM(),
    adaptive = false,
    dt = minimum(Δt) / 2,
    save_everystep = true,
    save_noise = true,
    trajectories = pars.nens
);
XanT = [ui.u[end] for ui in sol.u]
@time cvg = convergence(Δt, sol, XanT, oscillator_setd1!);

# %%
fig, ax = figax(nx = 1, h = 5, xscale = log2, yscale = log2, xlabel = L"h")
@unpack Δt, es, ew = cvg
scatterlines!(ax, Δt, es; markersize = 20, linestyle = :dash, label = "Strong")
scatterlines!(ax, Δt, ew; markersize = 20, linestyle = :dash, label = "Weak")
lines!(ax, Δt, @. 1.e-1 * Δt)
lines!(ax, Δt, @. 4.e-3 * Δt)
axislegend(position = :rb)
save("figs/convergence_etd1.pdf", fig)
fig

# %%
eprob = EnsembleProblem(prob)
sol = solve(
    eprob,
    EM(),
    adaptive = false,
    dt = minimum(Δt) / 2,
    save_everystep = true,
    save_noise = true,
    trajectories = pars.nens
);
XanT = [ui.u[end] for ui in sol.u]
@time cvg1 = convergence(Δt, sol, XanT, oscillator_EM!);
@time cvg2 = convergence(Δt, sol, XanT, oscillator_srk2!);
@time cvg4 = convergence(Δt, sol, XanT, oscillator_etd2rk!);

# %%
fig, ax = figax(nx = 2, h = 5, xscale = log10, yscale = log10)
for (cvg, label) in zip([cvg1, cvg2, cvg3, cvg4], ["EULER", "SRK2", "ETD2RK"])
    @unpack Δt, es, ew = cvg
    scatterlines!(ax[1], Δt, es; label, markersize = 20)
    scatterlines!(ax[2], Δt, ew; label, markersize = 20)
end
fig

# %% [markdown]
# ## Convergence for the linear SDE

# %%
Random.seed!(42)
pars = (; x0 = 0.5, tmax = 40.0, nens = 100, T = 1.0, α = 1.0, β = 1.0);
f_lin(u, p, t) = -p.α * u + p.β
g_lin(u, p, t) = sqrt(2 * p.T)

prob = SDEProblem(f_lin, g_lin, pars.x0, (0.0, pars.tmax), pars)
sol = solve(prob, EM(), adaptive = false, dt = 5.e-2, save_everystep = true, save_noise = true);

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
Δt = @. 1 / 2^(10:-1:2)
eprob = EnsembleProblem(prob)
sol = solve(
    eprob,
    EM(),
    adaptive = false,
    dt = minimum(Δt) / 2,
    save_everystep = true,
    save_noise = true,
    trajectories = pars.nens
);
XanT = [final_solution(linearsde_analytical!, ui.W.t, ui.W.W, prob.p) for ui in sol.u];

# %%
fig, ax = figax(nx = 1, h = 5, xscale = log2, yscale = log2, xlabel = L"h")

@time cvg = convergence(Δt, sol, XanT, linearsde_setd1!);
@unpack Δt, es, ew = cvg
scatterlines!(ax, Δt, es; markersize = 20, linestyle = :dash, label = "Strong")
scatterlines!(ax, Δt, ew; markersize = 20, linestyle = :dash, label = "Weak")

lines!(ax, Δt, @. 4.e-1 * Δt)
lines!(ax, Δt, @. 4.e-2 * Δt)
axislegend(position = :rb)
fig

# %%
fig, ax = figax(nx = 2, h = 5, xscale = log10, yscale = log10)

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
