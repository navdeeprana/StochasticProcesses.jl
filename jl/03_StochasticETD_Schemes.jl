# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: sohrab 1.10.10
#     language: julia
#     name: sohrab-1.10
# ---

# %%
# Imports and setup
import Pkg;
Pkg.activate(".");
using Revise, Printf, MathTeXEngine, CairoMakie, DataFrames, StatsBase, Random
includet("src/plotting.jl")
includet("src/brownian.jl")
includet("src/repeated_vector.jl")
includet("src/etd_factors.jl")
includet("src/sde_algorithms.jl")
includet("src/convergence.jl")
includet("src/oscillator.jl")
colors = Makie.wong_colors();
set_theme!(makietheme())
CairoMakie.enable_only_mime!("html")
Random.seed!(42);

# %%
pars = (; x0 = 0.0, tmax = 10, nens = 10000, T = 6.0, Γ = 5.0, b = 1.e-2, z = 3);

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

f(u, p, t) = -p.Γ * (u + p.b * u^p.z)
g(u, p, t) = sqrt(2 * p.Γ * p.T)

prob = SDEProblem(f, g, pars.x0, (0.0, pars.tmax), pars)
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
pars = (; x0 = 1.0, tmax = 5.0, nens = 100, T = 5.e-2, Γ = 1.0, b = 1.e-2, z = 3);

Δt = @. 1 / 2^(11:-1:2)

prob = SDEProblem(f, g, pars.x0, (0.0, pars.tmax), pars);
sol = solve(prob, EM(), adaptive = false, dt = 1.e-2, save_everystep = true, save_noise = true);

# %%
eprob = EnsembleProblem(prob)
sol = solve(eprob, EM(), adaptive = false, dt = minimum(Δt) / 2, save_everystep = true, save_noise = true, trajectories = pars.nens);
t, W = noise_from_sol(sol)
XanT = [ui.u[end] for ui in sol.u]
@time cvg = convergence(pars, Δt, t, W, XanT, oscillator_setd1!);

# %%
c, h = 1, 1
fac = (exp(c*h), expm1(c*h)/c, exp(c*h/2))

# %%
fig, ax = figax(nx = 1, h = 5, xscale = log2, yscale = log2, xlabel = L"h")
plot_convergence(fig, ax, cvg, 1.0, 1.0)
# save("figs/convergence_etd1.pdf", fig)
fig

# %% [markdown]
# ## Convergence for the GBM

# %%
pars = (x0 = 1.0, tmax = 2.0, a = 2.0, b = 1.0, nens = 20000, δ = 0.1)
@time cvg1 = convergence_gbm(pars, gbm_setd1!);
@time cvg2 = convergence_gbm(pars, gbm_setd1_milstein!);

# %%
fig, ax = figax(nx = 2, h = 5, xscale = log2, yscale = log2)
plot_convergence(fig, ax[1], cvg1, 0.5, 1.0)
plot_convergence(fig, ax[2], cvg2, 1.0, 1.0)
ax[1].title = "SETD1 for GBM"
ax[2].title = "SETD1 Milstein for GBM"
fig

# %%
pars = (x0 = 1.0, tmax = 2.0, a = 2.0, b = 1.0, nens = 1000, δ = 0.0)
t, W = brownian_motion(2.e-2, pars.tmax)

fig, ax = figax(nx = 2)
xt = gbm_analytical.(pars.x0, pars.a, pars.b, t, W)
lines!(ax[1], t, xt, color = :black)
lines!(ax[2], t, xt, color = :black)

x = algorithm_trajectory(gbm_milstein!, t, W, pars)
lines!(ax[1], t, x, color = :red)

x = algorithm_trajectory(gbm_setd1_milstein!, t, W, pars)
lines!(ax[2], t, x, color = :blue)

fig

# %%
t, W = brownian_motion(1.e-2, pars.tmax, pars.nens)

fig, ax = figax(h = 4)
xt = map(Wi -> gbm_analytical.(pars.x0, pars.a, pars.b, t, Wi), eachcol(W))
lines!(ax, t, mean(xt), color = :black)

x = map(Wi -> algorithm_trajectory(gbm_milstein!, t, Wi, pars), eachcol(W))
lines!(ax, t, mean(x), color = :red)
x = map(Wi -> algorithm_trajectory(gbm_setd1_milstein!, t, Wi, pars), eachcol(W))
lines!(ax, t, mean(x), color = :blue)
resize_to_layout!(fig)
fig

# %% [markdown]
# ## Convergence for the linear SDE

# %%
Random.seed!(42)
pars = (; x0 = 0.5, tmax = 10.0, nens = 500, a = -1.0, b = 1.0, c = 0.5);
f_lin(u, p, t) = p.a * u + p.b
g_lin(u, p, t) = p.c

prob = SDEProblem(f_lin, g_lin, pars.x0, (0.0, pars.tmax), pars)
sol = solve(prob, EM(), adaptive = false, dt = 5.e-2, save_everystep = true, save_noise = true);

# %%
t, W = sol.t, sol.W.W

fig, ax = figax(a = 3)
lines!(ax, t, sol.u; linewidth = 3)

x = linearsde_analytical(pars.x0, pars.a, pars.b, pars.c, t, W)
lines!(ax, t, x; color = :black)
fig

# %%
linearsde_analytical!(x, t, W, pars) = linearsde_analytical!(x, pars.x0, pars.a, pars.b, pars.c, t, W)
Δt = @. 1 / 2^(10:-1:2)
eprob = EnsembleProblem(prob)
sol = solve(eprob, EM(), adaptive = false, dt = minimum(Δt) / 4, save_everystep = true, save_noise = true, trajectories = pars.nens);
t, W = noise_from_sol(sol)
XanT = [final_solution(linearsde_analytical!, ui.W.t, ui.W.W, prob.p) for ui in sol.u];
@time cvg = convergence(pars, Δt, t, W, XanT, linearsde_setd1!);

# %%
fig, ax = figax(nx = 1, h = 5, xscale = log2, yscale = log2, xlabel = L"h")
plot_convergence(fig, ax, cvg, 1.0, 1.0)
fig

# %% [markdown]
# # Stability of the schemes

# %%
pars = (; x0 = 1.0, tmax = 5, nens = 10000, a = -3.0, b = sqrt(2));

# %%
t, W = brownian_motion(5.e-3, pars.tmax, pars.nens)
Xan = map(Wi -> gbm_analytical.(pars.x0, pars.a, pars.b, t, Wi), eachcol(W));

# %%
fig, ax = figax(nx = 2)
for i in 1:10
    lines!(ax[1], t, Xan[i], color = (:black, 0.2))
end
lines!(ax[1], t, mean(Xan))
lines!(ax[2], t, var(Xan))
lines!(ax[2], t, gbm_var.(pars.x0, pars.a, pars.b, t))
ax[1].limits = (0, 3, nothing, nothing)
ax[2].limits = (0, 3, nothing, nothing)
fig

# %%
isEMstable(p, Δt) = abs(1+Δt*p.a)^2 + Δt*abs(p.b)^2 < 1

# %%
fig, ax = figax(yscale = Makie.pseudolog10)
for Δt in [1.e-2, 1.e-1, 5.e-1]
    t, W = brownian_motion(Δt, pars.tmax, pars.nens)
    XT = map(Wi -> algorithm_trajectory(gbm_euler_maruyama!, t, Wi, pars), eachcol(W));
    l = @sprintf "Δt=%.2f, Stable=%s" Δt string(isEMstable(pars, Δt))
    lines!(ax, t, var(XT), label = l)
end
t = 0.0:1.e-3:pars.tmax
lines!(ax, t, gbm_var.(pars.x0, pars.a, pars.b, t), color = :black)
axislegend(ax)
fig

# %%
fig, ax = figax(yscale = Makie.pseudolog10)
for Δt in [1.e-2, 1.e-1, 5.e-1]
    t, W = brownian_motion(Δt, pars.tmax, pars.nens)
    parsδ = (; pars..., δ = 0.1)
    XT = map(Wi -> algorithm_trajectory(gbm_setd1!, t, Wi, parsδ), eachcol(W));
    l = @sprintf "Δt=%.2f" Δt
    lines!(ax, t, var(XT), label = l)
    if Δt > 2.e-1
        parsδ = (; pars..., δ = 0.1)
        XT = map(Wi -> algorithm_trajectory(gbm_setd1!, t, Wi, parsδ), eachcol(W));
        lines!(ax, t, var(XT), label = l)
    end
end
t = 0.0:1.e-3:pars.tmax
lines!(ax, t, gbm_var.(pars.x0, pars.a, pars.b, t), color = :black)
fig

# %%
OU_setd1!(x, t, W, p, δ) = setd1!(
    x, t, W, p,
    p -> p.k-δ,
    (x0, p) -> δ*x0,
    (x0, p) -> sqrt(2*p.D)
)
OU_EMvariance(k, D, h) = 2*D*h/(1-(1+k*h)^2)

function OU_SETDvariance(k, D, h, δ)
    c = (k-δ)
    fac = (exp(c*h), (exp(c*h)-1)/c, sqrt((exp(2*c*h)-1)/(2c)))
    return 2*D*fac[3]^2/(1 - (fac[1]+δ*fac[2])^2)
end

function OU2linearsde(p)
    (; x0, tmax, nens, k, D) = p
    return (; x0, tmax, nens, a = k, b = 0, c = sqrt(2D))
end
pars = (; x0 = 1.0, tmax = 5, nens = 20000, k = -2.0, D = 1.0);
parsEM = OU2linearsde(pars)

# %%
function varerr!(vs, xt)
    x = Iterators.partition(xt, length(xt)÷40)
    vi = [var(xi) for xi in x]
    push!(vs.m, mean(vi))
    push!(vs.s, std(vi))
end

Δt = [2.5e-3, 5.e-3, 1.e-2, 2.e-2, 4.e-2, 8.e-2, 1.6e-1]
var1 = (; m = Float64[], s = Float64[])
var2 = (; m = Float64[], s = Float64[])

for Δti in Δt
    t, W = brownian_motion(Δti, pars.tmax, pars.nens)
    xt = map(Wi -> final_solution(linearsde_euler_maruyama!, t, Wi, parsEM), eachcol(W));
    varerr!(var1, xt)
    xt = map(Wi -> final_solution(OU_setd1!, t, Wi, pars, -0.5), eachcol(W));
    varerr!(var2, xt)
end

# %%
function errorscatter!(ax, x, y, dy; kw...)
    p = scatter!(ax, x, y; kw...)
    errorbars!(ax, x, y, dy; color = p.color, whiskerwidth = 0.8*to_value(p.markersize)[1])
end

fig, ax = figax(xscale = log2)
errorscatter!(ax, Δt, var1.m, var1.s, label = "Euler")
errorscatter!(ax, Δt, var2.m, var2.s, label = "SETD1")
lines!(ax, Δt, OU_EMvariance.(pars.k, pars.D, Δt), color = :black)
lines!(ax, Δt, OU_SETDvariance.(pars.k, pars.D, Δt, -0.5), color = :black)
axislegend(ax, position = :lt)
resize_to_layout!(fig)
fig

# %%
