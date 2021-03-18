# Internal threaded random normal sampler
for T in (Float64, Float32, Float16)
    @eval function _randn(::Type{$T}, n::Int, d::Int)
        Z = SharedMatrix{$T}(n, d)
        @inbounds @threads for i in eachindex(Z)
            Z[i] = randn($T)
        end
        sdata(Z)
    end
    @eval _randn(::Type{$T}, n::Real, d::Real) = _randn($T, Int(n), Int(d))
end
_randn(n::Real, d::Real) = _randn(Float64, Int(n), Int(d))


# Internal random multivariate normal
for T in (Float64, Float32, Float16)
    @eval function _rmvn(n::Int, ρ::Matrix{$T})
        Z = _randn($T, n, size(ρ, 1))
        C = cholesky(ρ)
        Z * Matrix{$T}(C.U)
    end
end
_rmvn(n::Int, ρ::Float64) = _rmvn(n, [1.0 ρ; ρ 1.0])


normal_to_margin(d::UD, x::Float64) = quantile(d, _normcdf(x))
normal_to_margin(d::UD, x::Real) = normal_to_margin(d, Float64(x))
normal_to_margin(d::UD, A::Array{<:Real, N}) where {N} = normal_to_margin.(d, Array{Float64, N}(A))


"""
    rmvn(n[, μ], Σ)

Fast parrallel generation of multivariate normal samples.
"""
function rmvn end
for T in (Float64, Float32, Float16)
    @eval function rmvn(n::Int, μ::Vector{$T}, Σ::Matrix{$T})
        μ' .+ _rmvn(n, Σ)
    end
    @eval rmvn(n::Real, μ::Vector{$T}, Σ::Matrix{$T}) = rmvn(Int(n), μ, Σ)
    @eval function rmvn(n::Real, Σ::Matrix{$T})
        d = size(Σ, 2)
        rmvn(Int(n), zeros($T, d), Σ)
    end
end
function rmvn(n::Real, μ::Vector{<:Real}, Σ::Matrix{<:Real})
    rmvn(Int(n), Vector{Float64}(μ), Matrix{Float64}(Σ))
end


"""
    rvec(n, ρ::Matrix, margins::Vector{<:UnivariateDistribution})

Generate samples for a list of marginal distributions and a correaltion structure.

# Examples
```julia-repl
julia> using Distributions

julia> import LinearAlgebra: diagind

julia> margins = [Normal(3, 1), LogNormal(3, 1), Exponential(3)]

julia> R = fill(0.5, 3, 3); r[diagind(r)] .= 1.0;

julia> rvec(10, R, margins)
#>10×3 Array{Float64,2}:
 3.71109  82.0696   8.14332
 1.23022  38.8599   2.64595
 2.07222   3.76843  1.13465
 2.82434  11.6953   0.891066
 2.37599  10.0552   1.80555
 2.85431  25.4935   3.40865
 3.85253  21.2241   3.67532
 3.70605  59.2439   2.02502
 2.34066   1.89257  0.619948
 3.83507  16.787    0.66837
```
"""
function rvec end
for T in (Float64, Float32, Float16)
    @eval function rvec(n::Int, ρ::Matrix{$T}, margins::Vector{<:UD})
        d = length(margins)
        r,s = size(ρ)

        !(r == s == d) && throw(DimensionMismatch("The number of margins must match the size of the correlation matrix."))
        !iscorrelation(ρ) && throw(ValidCorrelationError())

        Z = SharedMatrix{$T}(_rmvn(n, ρ))
        @inbounds @threads for i in 1:d
            Z[:,i] = normal_to_margin(margins[i], Z[:,i])
        end
        sdata(Z)
    end
    @eval rvec(n::Real, ρ::Matrix{$T}, margins::Vector{<:UD}) = rvec(Int(n), ρ, margins)
end
