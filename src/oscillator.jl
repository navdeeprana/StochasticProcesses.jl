function euler_maruyama!(x, t, W, pars, fhD, frhs)
    h, D = fhD(t, pars)
    fη = sqrt(2 * D)
    x[1] = pars.x0
    @inbounds for i in 2:length(W)
        x[i] = x[i-1] + h * frhs(x[i-1], pars) + fη * (W[i] - W[i-1])
    end
end

function oscillator_EM!(x, t, W, p)
    euler_maruyama!(
        x, t, W, p,
        (t, p) -> (t[2] - t[1], p.Γ * p.T),
        (x, p) -> -p.Γ * (x + p.b * x^3)
    )
end

linearsde_EM!(x, t, W, p) = euler_maruyama!(
    x, t, W, p,
    (t, p) -> (t[2] - t[1], p.T),
    (x, p) -> -p.α * x + p.β
)

function setd1!(x, t, W, pars, fchD, fnlin)
    c, h, D = fchD(t, pars)
    f = (etd1_factors(h, c)..., etd_stochastic(h, c, D)/sqrt(h))
    x[1] = pars.x0
    @inbounds for i in 2:length(W)
        x[i] = f[1] * x[i-1] + f[2] * fnlin(x[i-1], pars) + f[3] * (W[i] - W[i-1])
    end
end

oscillator_setd1!(x, t, W, p) = setd1!(
    x, t, W, p,
    (t, p) -> (-p.Γ, t[2] - t[1], p.Γ * p.T),
    (x, p) -> -p.Γ * p.b * x^3
)

linearsde_setd1!(x, t, W, p) = setd1!(
    x, t, W, p,
    (t, p) -> (-p.α, t[2] - t[1], p.T),
    (x, p) -> p.β
)

function etd2rk_factors_oscillator(h, pars)
    @unpack Γ, b, T = pars
    c, D = -Γ, Γ * T
    f = etd2rk_factors(h, c)
    return (f[1], -Γ * b * f[2], -Γ * b * f[3], etd_stochastic(h, c, D))
end

function oscillator_euler_ensemble(pars, Δt)
    @unpack nens, tmax, Γ, b, T = pars
    iters = round(Int, tmax / Δt)
    fη = sqrt(2 * Γ * T * Δt)
    x, η = zeros(nens), zeros(nens)
    for t in 1:iters
        randn!(η)
        @. x = x + Δt * (-Γ * (x + b * x^3)) + fη * η
    end
    return x
end

function oscillator_etd1_ensemble(pars, Δt)
    @unpack nens, tmax, Γ, b, T = pars
    iters = round(Int, tmax / Δt)
    c, h, D = -pars.Γ, Δt, pars.Γ * pars.T
    f = (etd1_factors(h, c)..., etd_stochastic(h, c, D))
    x, η = zeros(nens), zeros(nens)
    for t in 1:iters
        randn!(η)
        @. x = f[1] * x - pars.Γ * pars.b * f[2] * x^3 + f[3] * η
    end
    return x
end

function auxvars(t, pars)
    @unpack Γ, b, T = pars
    Δt = t[2] - t[1]
    fη = sqrt(2 * pars.Γ * pars.T * Δt)
    return Γ, b, T, Δt, fη
end

function oscillator_euler!(x, t, W, pars)
    Γ, b, T, Δt, fη = auxvars(t, pars)
    x[1] = pars.x0
    @inbounds for i in 2:length(W)
        x[i] = x[i-1] - Δt * Γ * (x[i-1] + b * x[i-1]^3) + fη * (W[i] - W[i-1]) / sqrt(Δt)
    end
end

function oscillator_etd2rk!(x, t, W, pars)
    Δt = t[2] - t[1]
    f = etd2rk_factors_oscillator(t[2] - t[1], pars)
    x[1] = pars.x0
    @inbounds for i in 2:length(W)
        y = f[1] * x[i-1] + f[2] * x[i-1]^3 + f[4] * (W[i] - W[i-1]) / sqrt(Δt)
        x[i] = y + f[3] * (y^3 - x[i-1]^3)
    end
end

function oscillator_srk2!(x, t, W, pars)
    @inline _det(x, pars) = -pars.Γ * (x + pars.b * x^3)
    Γ, b, T, Δt, fη = auxvars(t, pars)
    x[1] = pars.x0
    @inbounds for i in 2:length(W)
        η = (W[i] - W[i-1]) ./ sqrt(Δt)
        f1 = _det(x[i-1], pars)
        f2 = _det(x[i-1] + Δt * f1 + fη * η, pars)
        x[i] = x[i-1] + 0.5 * Δt * (f1 + f2) + fη * η
    end
end

function OU_analytical!(x, t, W, pars)
    @assert pars.b == 0.0 "b=0 for analytical solution."
    Γ, b, T, Δt, fη = auxvars(t, pars)

    x[1] = pars.x0
    ito = 0.0
    for i in 2:length(W)
        ito = ito + exp(Γ * t[i]) * (W[i] - W[i-1]) / sqrt(Δt)
        x[i] = exp(-Γ * t[i]) * (pars.x0 + fη * ito)
    end
    nothing
end

function final_solution(algo::F, args...; kwargs...) where {F}
    x = SEVector()
    algo(x, args...; kwargs...)
    return x.v[1]
end
