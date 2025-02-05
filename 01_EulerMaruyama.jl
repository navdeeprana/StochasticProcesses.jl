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

# # 1. Euler-Maruyama algorithm

# +
gbm_analytical(λ, μ, t, W) = exp((λ - μ^2 / 2) * t + μ * W)

function gbm_euler_maruyama(pars, t, W)
    @unpack λ, μ = pars
    Δt = t[2] - t[1]
    X = zeros(length(W))
    X[1] = 1
    @inbounds for i in 2:length(W)
        Xp, dW = X[i-1], W[i] - W[i-1]
        X[i] = Xp + λ * Δt * Xp + μ * Xp * dW
    end
    return X
end

function gbm_meta(pars, Δt)
    @unpack tmax, λ, μ = pars
    t, W = brownian_motion(Δt, tmax)
    X = gbm_euler_maruyama(pars, t, W)
    Xan = gbm_analytical.(λ, μ, t, W)
    return t, W, X, Xan
end
# -

# ## HW : Solve the GBM analytically

# +
pars = (tmax = 2.0, λ = 2.0, μ = 1.0)
fig, ax = figax(h = 5, nx = 2, ny = 2, xlabel = "t", ylabel = "W(t)")

for (axi, Δt) in zip(ax, [2.e-1, 1.e-1, 1.e-2, 1.e-3])
    t, W, X, Xan = gbm_meta(pars, Δt)
    lines!(axi, t, Xan, label = "Analytical")
    lines!(axi, t, X, label = "Euler-Maruyama")
    axi.title = @sprintf "GBM Δt = %.3f" Δt
    axislegend(axi, position = :lt)
end
fig
# -

# # 2. Strong and weak convergence

# +
function gbm_euler_maruyama_final(pars, t, W)
    @unpack λ, μ = pars
    Δt = t[2] - t[1]
    X = 1
    @inbounds for i in 2:length(W)
        X += λ * Δt * X + μ * X * (W[i] - W[i-1])
    end
    return X
end

function convergence(pars, gbm_discrete::F) where {F}
    @unpack nens, tmax, λ, μ = pars

    Δt = @. 1 / 2^(5:10)
    N = @. round(Int, tmax / Δt)
    t, W = brownian_motion(Δt[end], tmax, nens)

    Wend = collect(W[end, :])
    Xan_end = gbm_analytical.(λ, μ, tmax, Wend)

    estrong, eweak = Float64[], Float64[]
    for Ni in N
        skip = N[end] ÷ Ni
        @views tn, Wn = t[1:skip:end], W[1:skip:end, :]

        Xend = map(Wni -> gbm_discrete(pars, tn, Wni), eachcol(Wn))
        push!(estrong, mean(@. abs(Xan_end - Xend)))
        push!(eweak, abs(mean(Xan_end) - mean(Xend)))
    end
    return N, estrong, eweak
end
# -

pars = (nens = 10000, tmax = 2.0, λ = 2.0, μ = 1.0)
fig, ax = figax(h = 5, xscale = log2, yscale = log2)
N, es, ew = convergence(pars, gbm_euler_maruyama_final)
plot_convergence(fig, ax, N, es, ew, -0.5, -1.0)
ax.title = "Convergence for Euler-Maruyama algorithm"
fig

# # 3. Milstein algorithm

function gbm_milstein_final(pars, t, W)
    @unpack λ, μ = pars
    Δt = t[2] - t[1]
    X = 1
    @inbounds for i in 2:length(W)
        dW = W[i] - W[i-1]
        X += λ * Δt * X + μ * X * dW + 0.5 * μ^2 * X * (dW^2 - Δt)
    end
    return X
end

# ## HW : Implement Milstein algorithm for the full trajectory

fig, ax = figax(h = 5, xscale = log2, yscale = log2)
N, es, ew = convergence(pars, gbm_milstein_final)
plot_convergence(fig, ax, N, es, ew, -1.0, -1.0)
ax.title = "Convergence for Milstein algorithm"
fig

# What to do when analytical solution is not known?
function convergence_noanalytical(pars, algo::F1, algo_true::F2; scale = 4) where {F1,F2}
    @unpack nens, tmax, λ, μ = pars

    Δt = @. 1 / 2^(5:10)
    N = @. round(Int, tmax / Δt)
    t, W = brownian_motion(Δt[end] / scale, tmax, nens)

    Xan_end = map(Wi -> algo_true(pars, t, Wi), eachcol(W))

    estrong, eweak = Float64[], Float64[]
    for Ni in N
        skip = length(t) ÷ Ni
        @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
        Xend = map(Wni -> algo(pars, tn, Wni), eachcol(Wn))

        push!(estrong, mean(@. abs(Xan_end .- Xend)))
        push!(eweak, abs(mean(Xan_end) - mean(Xend)))
    end
    return N, estrong, eweak
end

fig, ax = figax(nx = 3, h = 5, xscale = log2, yscale = log2)
for (scale, axi) in zip([2, 4, 8], ax)
    N, es, ew = convergence_noanalytical(pars, gbm_euler_maruyama_final, gbm_euler_maruyama_final; scale)
    plot_convergence(fig, axi, N, es, ew, -0.5, -1.0)
    axi.title = "scale = $scale"
end
fig

fig, ax = figax(nx = 3, h = 5, xscale = log2, yscale = log2)
for (scale, axi) in zip([1, 2, 4], ax)
    N, es, ew = convergence_noanalytical(pars, gbm_euler_maruyama_final, gbm_milstein_final; scale)
    plot_convergence(fig, axi, N, es, ew, -0.5, -1.0)
    axi.title = "scale = $scale"
end
fig

# # 4. StochasticDiffEq.jl

# +
using StochasticDiffEq

pars = (tmax = 2.0, λ = 2, μ = 1)
tspan, u0 = (0.0, pars.tmax), 1.0

f(u, p, t) = p.λ * u
g(u, p, t) = p.μ * u

prob = SDEProblem(f, g, u0, tspan, pars)
sol = solve(prob, EM(), dt = 1.e-3);
# -

fig, ax = figax(xlabel = "time", ylabel = "X(t)")
lines!(ax, sol.t, sol.u)
fig

# +
u_analytic(u0, p, t, W) = u0 * exp((p.λ - p.μ^2 / 2) * t + p.μ * W)
func = SDEFunction(f, g, analytic = u_analytic)
prob = SDEProblem(func, u0, tspan, pars)
sol = solve(prob, EM(), dt = 1.e-3)

fig, ax = figax(title = "Comparison with analytics", xlabel = "time", ylabel = "X(t)")
lines!(ax, sol.t, sol.u, label = "EM()")
lines!(ax, sol.t, sol.u_analytic, label = "Analytical")
axislegend(ax, position = :lt)
fig
# -

sol = solve(prob, SRIW1())
fig, ax = figax(title = "Higher order methods")
lines!(ax, sol.t, sol.u, label = "SRIW1()")
lines!(ax, sol.t, sol.u_analytic, label = "Analytical")
axislegend(ax, position = :lt)
fig

# Ensemble
eprob = EnsembleProblem(prob)
sol = solve(eprob, EM(), dt = 1.e-2, trajectories = 4)
fig, ax = figax(title = "Ensemble of problems")
for ui in sol.u
    lines!(ax, ui.t, ui.u)
end
fig

function convergence_dejl(prob, nens, algorithm)
    Δt = @. 1 / 2^(5:10)
    N = @. round(Int, prob.p.tmax / Δt)
    estrong, eweak = Float64[], Float64[]
    eprob = EnsembleProblem(prob)
    for Δti in Δt
        sol = solve(eprob, algorithm, adaptive = false, dt = Δti, trajectories = nens)
        XT = [ui.u[end] for ui in sol.u]
        XanT = [ui.u_analytic[end] for ui in sol.u]
        push!(estrong, mean(@. abs(XanT .- XT)))
        push!(eweak, abs(mean(XanT) - mean(XT)))
    end
    return N, estrong, eweak
end

cvg = (EM = convergence_dejl(prob, 5000, EM()), SRIW1 = convergence_dejl(prob, 5000, SRIW1()));

fig, ax = figax(nx = 2, h = 5, xscale = log2, yscale = log2)
plot_convergence(fig, ax[1], cvg.EM..., -0.5, -1.0)
ax[1].title = "EM()"
plot_convergence(fig, ax[2], cvg.SRIW1..., -1.5, -2.0)
ax[2].title = "SRIW1()"
fig

# ## HW : Play with various other algorithms in StochasticDiffEq.jl, check their convergence


