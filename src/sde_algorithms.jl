# Some common SDE examples and algorithms.
# Equations are described in Kloeden 1999 Section 4.4 and the code follows the same notation.

############## Linear SDEs with Additive Noise (Equation 4.2) ##############
# dx = ax(t)dt + bdt + cdW

function linearsde_analytical!(x, x0, a, b, c, t, W)
    x[1] = x0
    ito = 0.0
    for i in 2:length(W)
        at = a*t[i]
        ito = ito + exp(-at) * (W[i] - W[i-1])
        x[i] = exp(at) * (x0 + (b/a) * (1 - exp(-at)) + c * ito)
    end
    nothing
end

function linearsde_analytical(x0, a, b, c, t, W)
    x = zero(W)
    linearsde_analytical!(x, x0, a, b, c, t, W)
    return x
end

############## OU process (maps to 4.2 with a=-k, b=0, and c=sqrt(2D)) ##############
# dx = -kx(t)dt + sqrt(2D)dW

OU_analytical(x0, k, D, t, W) = linearsde_analytical(x0, -k, 0, sqrt(2D), t, W)

############## Geometric Brownian Motion (Equation 4.6) ##############
# dx = ax(t)dt + bx(t)dW

gbm_analytical(x0, a, b, t, W) = x0 * exp((a - b^2 / 2) * t + b * W)
gbm_analytical(a, b, t, W) = gbm_analytical(1, a, b, t, W)

gbm_var(x0, a, b, t) = x0^2*exp(2*a*t)*(exp(b^2*t)-1)

############## Algorithms ##############

# Note that W has sqrt(h) scaling inside.

function euler_maruyama!(x, t, W, p, f::F, g::G) where {F,G}
    h = t[2] - t[1]
    x[1] = p.x0
    @inbounds for i in 2:length(W)
        x0, dW = x[i-1], W[i] - W[i-1]
        x[i] = x0 + h * f(x0, p) + g(x0, p) * dW
    end
    nothing
end

function milstein!(x, t, W, p, f::F, g::G, dg::DG) where {F,G,DG}
    h = t[2] - t[1]
    x[1] = p.x0
    @inbounds for i in 2:length(W)
        x0, dW = x[i-1], W[i] - W[i-1]
        x[i] = (
            x0 + h * f(x0, p)
            + g(x0, p) * (dW + 0.5*dg(x0, p)*(dW^2-h))
        )
    end
    nothing
end

# SETDs require du = cu + f(u,t) + g(u,t) dW form.

function setd1!(x, t, W, p, fc::FC, f::F, g::G) where {FC,F,G}
    h = t[2] - t[1]
    c = fc(p)
    fac = (exp(c*h), expm1(c*h)/c, sqrt(expm1(2*c*h)/(2c))/sqrt(h))
    x[1] = p.x0
    @inbounds for i in 2:length(W)
        x0, dW = x[i-1], W[i] - W[i-1]
        x[i] = fac[1] * x0 + fac[2] * f(x0, p) + fac[3] * g(x0, p) * dW
    end
    nothing
end

function setd_em!(x, t, W, p, fc::FC, f::F, g::G) where {FC,F,G}
    h = t[2] - t[1]
    c = fc(p)
    fac = (exp(c*h), expm1(c*h)/c, exp(c*h/2))
    x[1] = p.x0
    @inbounds for i in 2:length(W)
        x0, dW = x[i-1], W[i] - W[i-1]
        x[i] = fac[1] * x0 + fac[2] * f(x0, p) + fac[3] * g(x0, p) * dW
    end
    nothing
end

function setd1_milstein!(x, t, W, p, fc::FC, f::F, g::G, dg::DG) where {FC,F,G,DG}
    h = t[2] - t[1]
    c = fc(p)
    fac = (exp(c*h), expm1(c*h)/c, exp(c*h/2))
    x[1] = p.x0
    @inbounds for i in 2:length(W)
        x0, dW = x[i-1], W[i] - W[i-1]
        x[i] = (
            fac[1] * x0 + fac[2] * f(x0, p)
            + fac[3] * g(x0, p) * (dW + 0.5 * dg(x0, p) * (dW^2-h))
        )
    end
    nothing
end

linearsde_euler_maruyama!(x, t, W, p) = euler_maruyama!(
    x, t, W, p,
    (x, p) -> p.a * x + p.b,
    (x, p) -> p.c
)

linearsde_setd1!(x, t, W, p) = setd1!(
    x, t, W, p,
    p -> p.a,
    (x, p) -> p.b,
    (x, p) -> p.c
)

function gbm_euler_maruyama!(X, t, W, p)
    euler_maruyama!(
        X, t, W, p,
        (x0, p) -> p.a * x0,
        (x0, p) -> p.b * x0
    )
end

function gbm_milstein!(X, t, W, p)
    milstein!(
        X, t, W, p,
        (x0, p) -> p.a * x0,
        (x0, p) -> p.b * x0,
        (x0, p) -> p.b
    )
end

function gbm_setd1!(X, t, W, p)
    setd1!(
        X, t, W, p,
        p -> p.a-p.δ,
        (x0, p) -> p.δ*x0,
        (x0, p) -> p.b*x0
    )
end

function gbm_setd1_milstein!(X, t, W, p)
    setd1_milstein!(
        X, t, W, p,
        p -> p.a-p.δ,
        (x0, p) -> p.δ*x0,
        (x0, p) -> p.b*x0,
        (x0, p) -> p.b
    )
end

function algorithm_trajectory(algorithm, t, W, p)
    x = zero(W)
    algorithm(x, t, W, p)
    return x
end

function final_solution(algorithm::F, args...; kwargs...) where {F}
    x = RepeatedVector()
    algorithm(x, args...; kwargs...)
    return x.v[1]
end
