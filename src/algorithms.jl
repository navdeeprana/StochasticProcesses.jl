# Note that W has sqrt(h) scaling inside.

function euler_maruyama!(x, t, W, p, f::F, g::G) where {F,G}
    h = t[2] - t[1]
    x[1] = p.x0
    @inbounds for i in 2:length(W)
        x0, dW = x[i-1], W[i] - W[i-1]
        x[i] = x0 + h * f(x0, p) + g(x0, p) * dW
    end
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
end

# SETDs require du = cu + f(u,t) + sqrt(2D) g(u,t) dW form.
function setd1!(x, t, W, p, fcD::FCD, f::F, g::G) where {FCD,F,G}
    h = t[2] - t[1]
    c, D = fcD(p)
    fac = (etd1_factors(h, c)..., etd_stochastic(h, c, D)/sqrt(h))
    x[1] = p.x0
    @inbounds for i in 2:length(W)
        x0, dW = x[i-1], W[i] - W[i-1]
        x[i] = fac[1] * x0 + fac[2] * f(x0, p) + fac[3] * g(x0, p) * dW
    end
end

linearsde_EM!(x, t, W, p) = euler_maruyama!(
    x, t, W, p,
    (x, p) -> p.a * x + p.b,
    (x, p) -> p.c
)

linearsde_setd1!(x, t, W, p) = setd1!(
    x, t, W, p,
    p -> (p.a, p.c^2/2),
    (x, p) -> p.b,
    (x, p) -> 1
)

function final_solution(algo::F, args...; kwargs...) where {F}
    x = SEVector()
    algo(x, args...; kwargs...)
    return x.v[1]
end
