# Create a makie theme for plotting.
function makietheme()
    Theme(
        fontsize = 10,
        Axis = (
            backgroundcolor = :transparent,
            xgridvisible = false,
            ygridvisible = false,
            xlabelpadding = 3,
            ylabelpadding = 3,
            titlefont = :regular
        ),
        Lines = (linewidth = 2.0,),
        Scatter = (markersize = 20,)
    )
end

# Create a grid of axis to plot into.
function figax(; nx = 1, ny = 1, h = 5, a = 1.6, s = 100, fontsize = 24, sharex = false, sharey = false, kwargs...)
    (a > 1) ? size = (a * s * h * nx, s * h * ny) : size = (s * h * nx, s * h * ny / a)
    fig = Figure(; size = round.(Int, size), fontsize)
    ax = [Axis(fig[j, i]; aspect = AxisAspect(a), kwargs...) for i in 1:nx, j in 1:ny]
    for i in 1:nx
        colsize!(fig.layout, i, Aspect(1, a))
    end
    resize_to_layout!(fig)
    (nx * ny == 1) ? ax = ax[1] : nothing
    return fig, ax
end

# General power law
power_law(x, p, a) = @. a * (x / x[1])^p

function plot_convergence(fig, ax, N, es, ew, ps, pw)
    scatterlines!(ax, N, es, label = "Strong", markersize = 20, linestyle = :dash)
    scatterlines!(ax, N, ew, label = "Weak", markersize = 20, linestyle = :dash)
    lines!(ax, N, power_law(N, ps, es[1]), color = :black)
    lines!(ax, N, power_law(N, pw, ew[1]), color = :black)
    axislegend(ax, position = :lb)
    ax.xlabel = "N"
end

plot_probability_distribution!(ax, X; bins = 100, kw...) = stephist!(ax, X; normalization = :pdf, bins, kw...)

function plot_normal_distribution!(ax, xm; μ = 0.0, σ = 1.0, kw...)
    x = LinRange(-xm, xm, 1000)
    P = @. exp(-((x-μ)^2 / (2*σ^2))) / sqrt(2π*σ^2)
    lines!(ax, x, P; label = "Normal", kw...)
end