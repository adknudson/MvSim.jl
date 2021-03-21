"""
    cor_adjust(m::CorMat{<:Union{Spearman, Kendall}})
"""
cor_adjust(m::CorMat{<:SpKe}) = convert(CorMat{Adjusted}, m)


# case Pearson
function _pearson_match2(x::Real, dA::CUD, dB::CUD, n::Int)
    μA = mean(dA)
    μB = mean(dB)
    σA = std(dA)
    σB = std(dB)

    k = 0:1:n
    a = _get_hermite_coefs(dA, n)
    b = _get_hermite_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = c2 .* a[k+1] * b[k+1] * factorial(k)
    end
    coef[1] = c1 * c2 + c2 * a[1] * b[1] - x

    r = _solve_poly_pm_one(coef)

    length(r) == 1 && return r
    !isnan(r) && return _nearest_root(x, r)
    isnan(r) && warn("No root found") && return r
end
function _pearson_match2(x::Real, dA::DUD, dB::DUD, n::Int)
    maxA = maximum(dA)
    maxB = maximum(dB)
    maxA = isinf(maxA) ? quantile(dA, 0.99_999) : maxA
    maxB = isinf(maxB) ? quantile(dB, 0.99_999) : maxB

    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    minB = minimum(dB)

    # Support sets
    A = minA:maxA
    B = minB:maxB

    # z = Φ⁻¹[F(A)], α[0] = -Inf, β[0] = -Inf
    α = [-Inf; _norminvcdf.(cdf.(dA, A))]
    β = [-Inf; _norminvcdf.(cdf.(dB, B))]

    c2 = 1 / (σA * σB)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = _Gn0d(k, A, B, α, β, c2) / factorial(k)
    end
    coef[1] = -x

    r = _solve_poly_pm_one(coef)

    length(r) == 1 && return r
    !isnan(r) && return _nearest_root(x, r)
    isnan(r) && warn("No root found") && return r
end
function _pearson_match2(x::Real, dA::DUD, dB::CUD, n::Int)
    σA = std(dA)
    σB = std(dB)
    minA = minimum(dA)
    maxA = maximum(dA)

    maxA = isinf(maxA) ? quantile(dA, 0.99_999) : maxA

    A = minA:maxA
    α = [-Inf; _norminvcdf.(cdf.(dA, A))]

    c2 = 1 / (σA * σB)

    coef = zeros(Float64, n+1)
    for k in 1:n
        coef[k+1] = _Gn0m(k, A, α, dB, c2) / factorial(k)
    end
    coef[1] = -x

    r = _solve_poly_pm_one(coef)

    length(r) == 1 && return r
    !isnan(r) && return _nearest_root(x, r)
    isnan(r) && warn("No root found") && return r
end
_pearson_match2(x::Real, dA::CUD, dB::DUD, n::Int) = _pearson_match2(x, dB, dA, n)


function _pearson_match(x::Real, dA::UD, dB::UD; kwargs...)
    convert_to_gsdist = Bool(get(kwargs, :use_gsdist, true))
    hermite_degree = Int(get(kwargs, :hermite_degree, 7))

    if convert_to_gsdist
        msg = "The option to convert discrete distributions to a Generalized S-Distribution " *
        "is enabled (`use_gsdist=true`). This can increase computational efficiency with a " *
        "potential loss for accuracy."
        @warn msg maxlog=1

        cutoff = 200

        if typeof(dA) <: DUD
            maxA = maximum(dA)
            if isinf(maxA) maxA = quantile(dA, 0.99_999) end
            if maxA > cutoff
                dA = GSDist(dA)
            end
        end

        if typeof(dB) <: DUD
            maxB = maximum(dB)
            if isinf(maxB) maxB = quantile(dB, 0.99_999) end
            if maxB > cutoff
                dB = GSDist(dB)
            end
        end
    end

    l, u = cor_bounds(dA, dB, Pearson(); hermite_degree=hermite_degree)
    if x < l || x > u
        msg = @sprintf "Target correlation %.3f is outside of the theoretical bounds (%.3f, %.3f). Using the nearest bound as the target." x l u
        @warn msg
        x = clamp(x, l, u)
    end

    return _pearson_match2(x, dA, dB, hermite_degree)
end


"""
    cor_adjust(m::CorMat{Pearson}, margins::Vector{UnivariateDistribution}; kwargs...)
"""
function cor_adjust(m::CorMat{Pearson}, margins::Vector{UD}; kwargs...) end
