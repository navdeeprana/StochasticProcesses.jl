oscillator_EM!(x, t, W, p) = euler_maruyama!(
    x, t, W, p,
    (x, p) -> -p.Γ * (x + p.b * x^p.z),
    (x, p) -> sqrt(2*p.Γ*p.T)
)

oscillator_setd1!(x, t, W, p) = setd1!(
    x, t, W, p,
    p -> (-p.Γ, p.Γ * p.T),
    (x, p) -> -p.Γ * p.b * x^p.z,
    (x, p) -> 1
)

function etd2rk_factors_oscillator(h, pars)
    @unpack Γ, b, T = pars
    c, D = -Γ, Γ * T
    f = etd2rk_factors(h, c)
    return (f[1], -Γ * b * f[2], -Γ * b * f[3], etd_stochastic(h, c, D))
end

function oscillator_euler_ensemble(pars, Δt)
    @unpack nens, tmax, Γ, b, T, z = pars
    iters = round(Int, tmax / Δt)
    fη = sqrt(2 * Γ * T * Δt)
    x, η = fill(pars.x0, nens), zeros(nens)
    for t in 1:iters
        randn!(η)
        @. x = x + Δt * (-Γ * (x + b * x^z)) + fη * η
    end
    return x
end

function oscillator_etd1_ensemble(pars, Δt)
    @unpack nens, tmax, Γ, b, T, z = pars
    iters = round(Int, tmax / Δt)
    c, h, D = -pars.Γ, Δt, pars.Γ * pars.T
    f = (etd1_factors(h, c)..., etd_stochastic(h, c, D))
    x, η = fill(pars.x0, nens), zeros(nens)
    for t in 1:iters
        randn!(η)
        @. x = f[1] * x - pars.Γ * pars.b * f[2] * x^z + f[3] * η
    end
    return x
end

function auxvars(t, pars)
    @unpack Γ, b, T, z = pars
    Δt = t[2] - t[1]
    fη = sqrt(2 * pars.Γ * pars.T * Δt)
    return Γ, b, T, z, Δt, fη
end

function oscillator_euler!(x, t, W, pars)
    Γ, b, T, z, Δt, fη = auxvars(t, pars)
    x[1] = pars.x0
    @inbounds for i in 2:length(W)
        x[i] = x[i-1] - Δt * Γ * (x[i-1] + b * x[i-1]^z) + fη * (W[i] - W[i-1]) / sqrt(Δt)
    end
end

function oscillator_etd2rk!(x, t, W, pars)
    Δt = t[2] - t[1]
    f = etd2rk_factors_oscillator(t[2] - t[1], pars)
    x[1] = pars.x0
    @inbounds for i in 2:length(W)
        y = f[1] * x[i-1] + f[2] * x[i-1]^pars.z + f[4] * (W[i] - W[i-1]) / sqrt(Δt)
        x[i] = y + f[3] * (y^3 - x[i-1]^3)
    end
end

function oscillator_srk2!(x, t, W, pars)
    @inline _det(x, p) = -p.Γ * (x + p.b * x^p.z)
    Γ, b, T, z, Δt, fη = auxvars(t, pars)
    x[1] = pars.x0
    @inbounds for i in 2:length(W)
        η = (W[i] - W[i-1]) ./ sqrt(Δt)
        f1 = _det(x[i-1], pars)
        f2 = _det(x[i-1] + Δt * f1 + fη * η, pars)
        x[i] = x[i-1] + 0.5 * Δt * (f1 + f2) + fη * η
    end
end
