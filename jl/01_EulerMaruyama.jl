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
includet("src/repeated_vector.jl")
includet("src/sde_algorithms.jl")
includet("src/convergence.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
CairoMakie.enable_only_mime!("svg")
Random.seed!(42);

# %% [markdown]
# # 1. Euler-Maruyama algorithm

# %% [markdown]
# Following Higham, SIAM Review 43 (2001), we will illustrate various algorithms for the geometric Brownian motion, which is written as
# $$dX(t) = a X(t) + b X(t) dW(t).$$
#
# Staying consistent with the Kloeden (1999) notation, we use $a=\lambda$, and $b=\mu$.

# %%
function gbm_meta(pars, Δt)
    (; tmax, a, b) = pars
    t, W = brownian_motion(Δt, tmax)
    X = gbm_euler_maruyama(t, W, pars)
    Xan = gbm_analytical.(a, b, t, W)
    return t, W, X, Xan
end

# %% [markdown]
# ## HW : Solve the GBM analytically

# %%
pars = (x0 = 1.0, tmax = 2.0, a = 2.0, b = 1.0)
fig, ax = figax(h = 5, nx = 2, ny = 2, xlabel = "t", ylabel = "W(t)")

for (axi, Δt) in zip(ax, [2.e-1, 1.e-1, 1.e-2, 1.e-3])
    t, W, X, Xan = gbm_meta(pars, Δt)
    lines!(axi, t, Xan, label = "Analytical")
    lines!(axi, t, X, label = "Euler-Maruyama")
    axi.title = @sprintf "GBM Δt = %.3f" Δt
    axislegend(axi, position = :lt)
end
fig

# %% [markdown]
# # 2. Strong and weak convergence

# %%
pars = (x0 = 1.0, tmax = 2.0, a = 2.0, b = 1.0, nens = 20000)
fig, ax = figax(h = 5, xscale = log2, yscale = log2)
cvg = convergence_gbm(pars, gbm_euler_maruyama!)
plot_convergence(fig, ax, cvg, 0.5, 1.0; f = minimum)
ax.title = "Convergence for Euler-Maruyama algorithm"
fig

# %% [markdown]
# # 3. Milstein algorithm

# %% [markdown]
# ## HW : Implement Milstein algorithm for the full trajectory

# %%
fig, ax = figax(h = 5, xscale = log2, yscale = log2)
cvg = convergence_gbm(pars, gbm_milstein!)
plot_convergence(fig, ax, cvg, 1.0, 1.0)
ax.title = "Convergence for Milstein algorithm"
fig

# %%
# What to do when analytical solution is not known?
# Use numerical solution as true solution
function convergence_gbm_noanalytical(pars, algo::F1, algo_true::F2; scale = 4) where {F1,F2}
    (; nens, tmax, a, b) = pars
    Δt = @. 1 / 2^(5:10)
    t, W = brownian_motion(Δt[end] / scale, tmax, nens)
    XanT = map(Wi -> final_solution(algo_true, t, Wi, pars), eachcol(W))
    return convergence(pars, Δt, t, W, XanT, algo)
end

# %%
fig, ax = figax(nx = 3, h = 5, xscale = log2, yscale = log2)
for (scale, axi) in zip([2, 4, 8], ax)
    cvg = convergence_gbm_noanalytical(pars, gbm_euler_maruyama!, gbm_milstein!; scale)
    plot_convergence(fig, axi, cvg, 0.5, 1.0)
    axi.title = "scale = $scale"
end
fig

# %% [markdown]
# # 4. StochasticDiffEq.jl

# %%
# StochasticDiffEq has u as the variable.
using StochasticDiffEq

pars = (u0 = 1.0, tmax = 2.0, a = 2.0, b = 1.0, nens = 5000)

f(u, p, t) = p.a * u
g(u, p, t) = p.b * u

# Single realization
prob = SDEProblem(f, g, pars.u0, (0.0, pars.tmax), pars)
sol = solve(prob, EM(), dt = 1.e-3);

# Ensemble
eprob = EnsembleProblem(prob)
esol = solve(eprob, EM(), dt = 1.e-2, trajectories = 4);

# %%
fig, ax = figax(nx = 2, xlabel = "time", ylabel = "X(t)")
lines!(ax[1], sol.t, sol.u)

for ui in esol.u
    lines!(ax[2], ui.t, ui.u)
end
fig

# %%
# Higher order methods

u_analytic(u0, p, t, W) = gbm_analytical(u0, p.a, p.b, t, W)
func = SDEFunction(f, g, analytic = u_analytic)
prob = SDEProblem(func, pars.u0, (0.0, pars.tmax), pars)

sol = solve(prob, EM(), dt = 1.e-3)

fig, ax = figax(nx = 2, xlabel = "time", ylabel = "X(t)")
lines!(ax[1], sol.t, sol.u, label = "EM()")
lines!(ax[1], sol.t, sol.u_analytic, label = "Analytical")

sol = solve(prob, SRIW1())
lines!(ax[2], sol.t, sol.u, label = "SRIW1()")
lines!(ax[2], sol.t, sol.u_analytic, label = "Analytical")
axislegend.(ax, position = :lt)
fig

# %%
function convergence_dejl(prob, algorithm)
    Δt = @. 1 / 2^(5:10)
    N = @. round(Int, prob.p.tmax / Δt)
    eprob = EnsembleProblem(prob)
    kw = (; adaptive = false, save_everystep = true, save_noise = true, trajectories = prob.p.nens)

    es, ew = Float64[], Float64[]
    for Δti in Δt
        sol = solve(eprob, algorithm; dt = Δti, kw...)
        XT = [ui.u[end] for ui in sol.u]
        XanT = [ui.u_analytic[end] for ui in sol.u]
        _es_ew!(es, ew, XanT, XT)
    end
    return (; Δt, N, es, ew)
end

# %%
cvg = (EM = convergence_dejl(prob, EM()), SRIW1 = convergence_dejl(prob, SRIW1()));

# %%
fig, ax = figax(nx = 2, h = 5, xscale = log2, yscale = log2)
plot_convergence(fig, ax[1], cvg.EM, 0.5, 1.0)
ax[1].title = "EM()"
plot_convergence(fig, ax[2], cvg.SRIW1, 1.5, 2.0)
ax[2].title = "SRIW1()"
fig

# %% [markdown]
# ## HW : Play with various other algorithms in StochasticDiffEq.jl and check their convergence

# %%
# Kloeden 4.4.4

f(u, p, t) = (2 * u / (1 + t) + p.b * (1 + t)^2)
g(u, p, t) = p.b * (1 + t)^2
u_analytic(u0, p, t, W) = u0 * (1 + t)^2 + p.b * (1 + t)^2 * (t + W)

pars = (b = 1.0, u0 = 1.0, tmax = 1.0, nens = 1000)

func = SDEFunction(f, g, analytic = u_analytic)
prob = SDEProblem(func, pars.u0, (0.0, pars.tmax), pars)

sol = solve(prob, EM(), dt = 1.e-3);

# %%
fig, ax = figax(xlabel = "time", ylabel = "X(t)")
lines!(ax, sol.t, sol.u, label = "EM()")
lines!(ax, sol.t, sol.u_analytic, label = "Analytical")
axislegend(ax)
fig

# %%
cvg = (
    EM = convergence_dejl(prob, EM()),
    SRIW1 = convergence_dejl(prob, SRIW1())
);

# %%
fig, ax = figax(nx = 2, h = 5, xscale = log2, yscale = log2)
plot_convergence(fig, ax[1], cvg.EM, 1.0, 1.0)
ax[1].title = "EM()"
plot_convergence(fig, ax[2], cvg.SRIW1, 2.0, 2.0)
ax[2].title = "SRIW1()"
fig

# %%
# Extra (Convergence of ODE methods)

# %%
using OrdinaryDiffEq

function convergence_dejl_methods(prob, algorithm::F; scale = 4) where {F}
    Δt = @. 1 / 2^(4:9)
    N = @. round(Int, prob.p.tmax / Δt)

    sol = solve(prob, RK4(), adaptive = false, dt = Δt[end] / scale, saveat = prob.p.tmax)
    XanT = sol.u[end]

    es = Float64[]
    for dt in Δt
        soli = solve(prob, algorithm, adaptive = false, dt = dt, saveat = prob.p.tmax)
        XT = soli.u[end]
        push!(es, abs(XanT - XT))
    end
    return (; Δt, N, es)
end

# %%
f(u, p, t) = -p.Γ * (u + p.b * u^4)
pars = (; tmax = 1.0, T = 0.0, Γ = 1.0, b = 2.e-2);
prob = ODEProblem(f, 1.0, (0.0, pars.tmax), pars);

# %%
cvg = (
    Euler = convergence_dejl_methods(prob, Euler()),
    Midpoint = convergence_dejl_methods(prob, Midpoint()),
    AB3 = convergence_dejl_methods(prob, AB3()),
    KenCarp3 = convergence_dejl_methods(prob, KenCarp3())
);

# %%
fig, ax = figax(h = 5, xscale = log2, yscale = log2)
for (k, v) in pairs(cvg)
    scatterlines!(ax, v.Δt, v.es; label = String(k), markersize = 20)
end
axislegend(position = :rb)
fig
