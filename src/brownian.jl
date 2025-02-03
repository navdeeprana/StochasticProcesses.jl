# Sample a Brownian motion or a Wiener process from [0, tmax] with a time step Δt.
# Returns time and the Wiener process.
function brownian_motion(Δt, tmax)
    N = round(Int, tmax / Δt)
    t = Δt .* (0:1:N)
    dW = randn(N)
    W = zeros(N + 1)
    for i in 1:N
        W[i+1] = W[i] + sqrt(Δt) * dW[i]
    end
    return t, W
end

# Sample an ensemble of size nens of Brownian motions from [0, tmax] with a time step Δt.
# Returns time and a dataframe with all the Wiener processes.
function brownian_motion(Δt, tmax, nens)
    t, W1 = brownian_motion(Δt, tmax)
    df = DataFrame(W1 = W1)
    for e in 2:nens
        df[!, "W$e"] = brownian_motion(Δt, tmax)[2] # Drop the time
    end
    return t, df
end
