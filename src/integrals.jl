# Riemann integrals
function riemann_integral(f::F1, xpoint::F2, xs, xf, N) where {F1,F2}
    Δx = (xf - xs) / N
    sum = 0
    for i in 1:N
        xi = xpoint(xs, i, Δx)
        sum += f(xi)
    end
    return Δx * sum
end

leftpoint(xs, i, Δx) = xs + (i - 1) * Δx

riemann_integral_left(f, xs, xf, N) = riemann_integral(f, leftpoint, xs, xf, N)

midpoint(xs, i, Δx) = xs + (i - 1) * Δx + Δx / 2

riemann_integral_mid(f, xs, xf, N) = riemann_integral(f, midpoint, xs, xf, N)

# Stochastic integrals for the Wiener process W
function ito_integral(W)
    N = length(W)
    sum = 0
    for i in 1:N-1
        sum += W[i] * (W[i+1] - W[i])
    end
    return sum
end

function strato_integral(W, Δt)
    N = length(W)
    sum = 0
    for i in 1:N-1
        sum += 0.5 * (W[i+1] + W[i] + sqrt(Δt) * randn()) * (W[i+1] - W[i])
    end
    return sum
end

# One line variants of the above functions
@inbounds ito_oneliner(W) = sum(W[i] * (W[i+1] - W[i]) for i in 1:length(W)-1)
@inbounds strato_oneliner(W, Δt) =
    sum((W[i+1] + W[i] + sqrt(Δt) * randn()) * (W[i+1] - W[i]) / 2 for i in 1:length(W)-1)

# Exact stochastic integrals for the Wiener process
ito_exact(W, tmax) = W[end]^2 / 2 - tmax / 2
strato_exact(W) = W[end]^2 / 2
