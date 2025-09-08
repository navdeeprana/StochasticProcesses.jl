@inline function etd1_factors(h, c)
    f1 = exp(h * c)
    f2 = expm1(h * c) / c
    return f1, f2
end

@inline function etd2_factors(h, c)
    f1 = exp(h * c)
    f2 = ((1 + h * c) * expm1(h * c) - h * c) / (h * c^2)
    f3 = (-expm1(h * c) + h * c) / (h * c^2)
    return f1, f2, f3
end

@inline function etd2rk_factors(h, c)
    f1 = exp(h * c)
    f2 = expm1(h * c) / c
    f3 = (expm1(h * c) - h * c) / (h * c^2)
    return f1, f2, f3
end

@inline function etd1_factors_taylor(h, c)
    if (abs(h * c) <= 1.e-5)
        f1 = exp(h * c)
        f2 = h + 0.5e0 * c * h^2
        return f1, f2
    else
        return etd1_factors(h, c)
    end
end

@inline function etd2_factors_taylor(h, c)
    if (abs(h * c) <= 1.e-5)
        f1 = exp(h * c)
        f2 = 1.5e0 * h + (2.e0 / 3.e0) * c * h^2
        f3 = -0.5e0 * h - (1.e0 / 6.e0) * c * h^2
        return f1, f2, f3
    else
        return etd2_factors(h, c)
    end
end

# Stochastic ETD factor for the equation du/dt = c u + F(u,t) + sqrt(2D) η(t).
# For x < 0, expm1(x)/x > 0, thus there's no issue with the sqrt.
@inline etd_stochastic(h, c, D) = sqrt(D * expm1(2 * h * c) / c)

@inline function etd_stochastic_taylor(h, c, D)
    if (abs(h * c) <= 1.e-5)
        return sqrt(2 * D * h)
    else
        return etd_stochastic(h, c, D)
    end
end
