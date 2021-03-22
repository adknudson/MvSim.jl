"""
    rvec(n, ρ::Matrix, margins::Vector{UnivariateDistribution})

Generate samples for a list of marginal distributions and a correaltion structure.

# Examples
```julia-repl
julia> using Distributions

julia> import LinearAlgebra: diagind

julia> margins = [Normal(3, 1), LogNormal(3, 1), Exponential(3)]

julia> R = fill(0.5, 3, 3); r[diagind(r)] .= 1.0;
```
"""
function rvec end
for T in (Float64, Float32, Float16)
    @eval function rvec(n::Int, R::Matrix{$T}, margins::Vector{<:UD})
        d = length(margins)
        r,s = size(R)

        !(r == s == d) && throw(DimensionMismatch("The number of margins must match the size of the correlation matrix."))
        !iscorrelation(R) && throw(ValidCorrelationError())

        Z = SharedMatrix{$T}(_rmvn(n, R))
        @inbounds @threads for i in 1:d
            Z[:,i] = _normal_to_margin(margins[i], Z[:,i])
        end
        sdata(Z)
    end
    @eval rvec(n::Real, R::Matrix{$T}, margins::Vector{<:UD}) = rvec(Int(n), R, margins)
end
