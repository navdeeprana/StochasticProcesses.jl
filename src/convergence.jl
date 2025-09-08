function _es_ew!(es, ew, Xan, X)
    push!(es, mean(@. abs(Xan - X)))
    push!(ew, abs(mean(Xan) - mean(X)))
end

function convergence_gbm(pars, algorithm::F) where {F}
    (; nens, tmax, a, b) = pars
    Δt = @. 1 / 2^(5:10)
    t, W = brownian_motion(Δt[end], tmax, nens)
    XanT = gbm_analytical.(a, b, tmax, collect(W[end, :]))
    return convergence(pars, Δt, t, W, XanT, algorithm)
end

function convergence(pars, Δt, t, W, XanT, algorithm::F) where {F}
    N = @. round(Int, pars.tmax / Δt)
    es, ew = Float64[], Float64[]
    for Ni in N
        skip = (length(t) - 1) ÷ Ni
        @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
        XT = map(Wni -> final_solution(algorithm, tn, Wni, pars), eachcol(Wn))
        _es_ew!(es, ew, XanT, XT)
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
