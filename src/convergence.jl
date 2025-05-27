function noise_from_sol(sol)
    t, W = sol.u[1].W.t, DataFrame()
    for e in 1:length(sol.u)
        W[!, "W$e"] = sol.u[e].W.W
    end
    return t, W
end

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

function convergence(Δt, sol, XanT, algorithm::F) where {F}
    N = @. round(Int, prob.p.tmax / Δt)
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
