abstract type AbstractNumericalMethod end
struct EulerMaruyama <: AbstractNumericalMethod end
struct Milstein <: AbstractNumericalMethod end
struct SETDEulerMaruyama <: AbstractNumericalMethod end
struct SETDMilstein <: AbstractNumericalMethod end
struct SETD1 <: AbstractNumericalMethod end

struct Integrator{M<:AbstractNumericalMethod,Q}
    m::M # Integration method
    q::Q # Contains fixed parameters for the integration method.
end

EulerMaruyama(h) = Integrator(EulerMaruyama(), (h = h, sqrth = sqrt(h)))

Milstein(h) = Integrator(Milstein(), (h = h, sqrth = sqrt(h)))

# For left-point approximation, approx = 1.0 and for mid-point approximation, approx = 0.5.
# By default we choose mid-point approximation as it is more accurate.

function SETDEulerMaruyama(h, c, approx=0.5)
    fac = (exp(c*h), expm1(c*h)/c, exp(c*h*approx))
    Integrator(SETDEulerMaruyama(), (; h, sqrth = sqrt(h), fac))
end

function SETDMilstein(h, c, approx=0.5)
    fac = (exp(c*h), expm1(c*h)/c, exp(c*h*approx))
    Integrator(SETDMilstein(), (; h, sqrth = sqrt(h), fac))
end

function SETD1(h, c)
    fac = (exp(c*h), expm1(c*h)/c, sqrt(expm1(2*c*h)/(2c))/sqrt(h))
    Integrator(SETD1(), (; h, sqrth = sqrt(h), fac))
end

struct SODE{U,P}
    f::Function
    g::Function
    dg::Function
    p::P
end

stepforward(int::Integrator, s::SODE, u0, dW) = stepforward(int.m, int.q, s, u0, dW)

function stepforward(::EulerMaruyama, q, s::SODE, u0, dW)
    return u0 + q.h * s.f(u0, s.p) + s.g(u0, s.p) * dW
end

function stepforward(::Milstein, q, s::SODE, u0, dW)
    return (
        u0 + q.h * s.f(u0, s.p)
        + s.g(u0, s.p) * (dW + 0.5 * s.dg(u0, s.p) * (dW^2-q.h))
    )
end

function stepforward(::SETDEulerMaruyama, q, s::SODE, u0, dW)
    return q.fac[1] * u0 + q.fac[2] * s.f(u0, s.p) + q.fac[3] * s.g(u0, s.p) * dW
end

function stepforward(::SETDMilstein, q, s::SODE, u0, dW)
    return (
        q.fac[1] * u0
        + q.fac[2] * s.f(u0, s.p)
        + q.fac[3] * s.g(u0, s.p) * (dW + 0.5 * s.dg(u0, s.p) * (dW^2 - q.h))
    )
end

function stepforward(::SETD1, q, s::SODE, u0, dW)
    return q.fac[1] * u0 + q.fac[2] * s.f(u0, s.p) + q.fac[3] * s.g(u0, s.p) * dW
end

abstract type AbstractWeinerIncrement end

struct SampledWeinerIncrement{T} <: AbstractWeinerIncrement
    h::T
    sqrth::T
    dW::Vector{T}
end

function SampledWeinerIncrement(h::T) where {T}
    SampledWeinerIncrement{T}(h, sqrt(h), Float64[])
end

function SampledWeinerIncrement(h::T, tmax::T) where {T}
    dW = zeros(round(Int, tmax/h))
    randn!(dW)
    @. dW *= sqrt(h)
    SampledWeinerIncrement{T}(h, sqrt(h), dW)
end

function brownian_motion(dW::SampledWeinerIncrement)
    W = [0.0]
    Wi = 0.0
    for dWi in dW.dW
        Wi += dWi
        push!(W, Wi)
    end
    return W
end

struct InstantWeinerIncrement{T} <: AbstractWeinerIncrement
    h::T
    sqrth::T
end

function InstantWeinerIncrement(h::T) where {T}
    InstantWeinerIncrement{T}(h, sqrt(h))
end

struct ComputedWeinerIncrement{T} <: AbstractWeinerIncrement
    h::T
    sqrth::T
    W::Vector{T}
end

function ComputedWeinerIncrement(h::T, tmax::T) where {T}
    t, W = brownian_motion(h, tmax)
    ComputedWeinerIncrement{T}(h, sqrt(h), W)
end

Base.getindex(dW::SampledWeinerIncrement, i) = dW.dW[i]
Base.getindex(dW::InstantWeinerIncrement, i) = dW.sqrth * randn()
Base.getindex(dW::ComputedWeinerIncrement, i) = dW.W[i+1] - dW.W[i]

@inline _push!(::Val{true}, x, e) = push!(x, e)
@inline _push!(::Val{false}, x, e) = nothing

function solve(
    sde::SODE,
    int::Integrator,
    dW::AbstractWeinerIncrement,
    u0,
    tmax,
    saveat;
    save_after = 0.0,
    save_noise = false,
    save_initial = true
)
    (; h, sqrth) = int.q
    niters, nsave = @. round(Int, (tmax, saveat)/h)
    save_noise_val = Val(save_noise)
    sol = (; t = [0.0], u = [u0], W = [0.0])
    if !save_initial
        pop!(sol.t);
        pop!(sol.u);
        pop!(sol.W)
    end
    ui, Wi = u0, 0.0
    for i in 1:niters
        dWi = dW[i]
        Wi += dWi
        ui = stepforward(int, sde, ui, dWi)
        if (mod(i, nsave) == 0) && (i*h >= save_after)
            push!(sol.t, i*h)
            push!(sol.u, ui)
            _push!(save_noise_val, sol.W, Wi)
        end
    end
    return sol
end

function compute_convergence!(cvg, h, sol, sol_an, mean_an)
    for (i, ti) in enumerate(sol[1].t)
        ti == 0 ? continue : nothing
        u = [s.u[i] for s in sol]
        uan = [s[i] for s in sol_an]
        es = mean(@. abs(u - uan))
        ew = abs(mean(u) - mean(uan))
        ew_an = abs(mean(u) - mean_an[i])
        ci = 1.96 * std(u) / sqrt(length(sol))
        push!(cvg, (ti, h, es, ew, ew_an, ci))
    end
end

function tnWn(t, W, twhen)
    skip = (length(t) - 1) ÷ round(Int, t[end] / twhen)
    @views tn, Wn = t[1:skip:end], W[1:skip:end, :]
    return tn, Wn
end

function convergence(sde, integrator, pars, h_cvg, t, W, sol_an, mean_an)
    cvg = DataFrame(
        t = Float64[], h = Float64[], es = Float64[], ew = Float64[], ew_an = Float64[], ci = Float64[]
    )
    for h in h_cvg
        tn, Wn = tnWn(t, W, h)
        sol = map(
            Wni -> solve(
                sde, integrator(h),
                ComputedWeinerIncrement{typeof(h)}(h, sqrt(h), Wni),
                pars.u0, pars.tmax, pars.saveat; save_noise = true
            ),
            eachcol(Wn)
        )
        compute_convergence!(cvg, h, sol, sol_an, mean_an)
    end
    return cvg
end
