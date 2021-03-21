# Estimate the upper and lower correlation bounds
function _sim_and_sort(dA::UD, dB::UD, C::PeSpKe, n::Int)
    a = rand(dA, n)
    b = rand(dB, n)

    sort!(a)
    sort!(b)
    upper = cor(a, b, C)

    reverse!(b)
    lower = cor(a, b, C)

    return (lower = lower, upper = upper)
end
_sim_and_sort(dA::UD, dB::UD, C::PeSpKe, n::Real) = _sim_and_sort(dA, dB, C, Int(n))

# This method is more accurate, but much slower for discrete distributions
function _sim_quantile_complement(dA::UD, dB::UD, C::PeSpKe, n::Int)
    u = rand(Float64, n)
    x = quantile.(dA, u)
    y = quantile.(dB, u)
    yᶜ = cquantile.(dB, u)

    upper = cor(x, y, C)
    lower = cor(x, yᶜ, C)

    return (lower = lower, upper = upper)
end
_sim_quantile_complement(dA::UD, dB::UD, C::PeSpKe, n::Real) = _sim_quantile_complement(dA, dB, C, Int(n))

_sim_bounds(dA::CUD, dB::CUD, C::PeSpKe, n::Int) = _sim_quantile_complement(dA, dB, C, n)
_sim_bounds(dA::UD, dB::UD, C::PeSpKe, n::Int) = _sim_and_sort(dA, dB, C, n)
_sim_bounds(dA::UD, dB::UD, C::PeSpKe, n::Real) = _sim_bounds(dA, dB, C, Int(n))


function _pearson_bounds(dA::UD, dB::UD, n::Int)
    μA = mean(dA)
    σA = std(dA)
    μB = mean(dB)
    σB = std(dB)

    k = 0:1:n
    a = _get_hermite_coefs(dA, n)
    b = _get_hermite_coefs(dB, n)

    c1 = -μA * μB
    c2 = 1 / (σA * σB)
    kab = factorial.(k) .* a .* b
    ρ_l = c1 * c2 + c2 * sum((-1).^k .* kab)
    ρ_u = c1 * c2 + c2 * sum(kab)

    ρ_l, ρ_u = _cor_clamp.((ρ_l, ρ_u))
    (lower = ρ_l, upper = ρ_u)
end


"""
    cor_bounds(dA::UD, dB::UD, C::SpKe; n_samples=100_000)

Find the stochastic lower and upper Spearman or Kendall correlation bounds bounds between
two marginal distributions. `n_samples` is the number of random samples used to estimate the
correlation bounds.
"""
function cor_bounds(dA::UD, dB::UD, C::SpKe; n_samples=100_000, kwargs...)
    return _sim_bounds(dA, dB, C, n_samples)
end

"""
    cor_bounds(dA::UD, dB::UD, C::Pearson; kwargs...)

Find the lower and upper Pearson correlation bounds. See Arguments

# Arguments
- `n_samples::Int=100_000`: Number of random samples used to estimate the correlation bounds.
  This argument is ignored if `use_exact=true`.
- `use_exact::Bool=true`: If true (the default), then the bounds will be calculated using
  an exact method
- `hermite_degree::Int=7`:
"""
function cor_bounds(dA::UD, dB::UD, C::Pearson; kwargs...)
    use_exact = Bool(get(kwargs, :use_exact, true))
    hermite_degree = Int(get(kwargs, :hermite_degree, 7))
    n_samples = Int(get(kwargs, :n_samples, 100_000))

    # check if using exact method
     use_exact && return _pearson_bounds(dA, dB, hermite_degree)

    # if no key or use_exact=false, then use stochastic method
    return _sim_bounds(dA, dB, C, n_samples)
end

function cor_bounds(margins::Vector{UD}, C::PeSpKe=Pearson(); kwargs...)
    d = length(margins)
    lower, upper = zeros(Float64, d, d), zeros(Float64, d, d)

    @simd for i in collect(subsets(1:d, Val{2}()))
        l, u = cor_bounds(margins[i[1]], margins[i[2]], C; kwargs...)
        lower[i...] = l
        upper[i...] = u
    end

    lower .= Symmetric(lower)
    cor_constrain!(lower)

    upper .= Symmetric(upper)
    cor_constrain!(upper)

    (lower = lower, upper = upper)
end
