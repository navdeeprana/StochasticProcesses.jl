function convergence(pars, Δt, sol, XanT, algorithm::F) where {F}
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
    kw = (; adaptive = false, save_everystep = true, save_noise = true, trajectories = prob.p.nens)
    for Δti in Δt
        sol = solve(eprob, algorithm; dt = Δti, kw...)
        XT = [ui.u[end] for ui in sol.u]
        XanT = [final_solution(linearsde_analytical!, ui.W.t, ui.W.W, prob.p) for ui in sol.u]
        push!(es, mean(@. abs(XanT .- XT)))
        push!(ew, abs(mean(XanT) - mean(XT)))
    end
    return (; Δt, N, es, ew)
end

function noise_from_sol(sol)
    t, W = sol.u[1].W.t, DataFrame()
    for e in 1:length(sol.u)
        W[!, "W$e"] = sol.u[e].W.W
    end
    return t, W
end

function noise_increment_from_sol(sol, N)
    t, dW = sol.u[1].W.t[1:N:end], DataFrame()
    for e in 1:length(sol.u)
        Wi = sol.u[e].W.W[1:N:end]
        dWi = Wi[2:end] .- Wi[1:(end-1)]
        dW[!, "W$e"] = dWi
    end
    return t, dW
end
