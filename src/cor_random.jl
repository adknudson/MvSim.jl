"""
    cor_rand([T::Type{<:AbstractFloat}], d::Int[, k::Int=d-1])

Return a random positive semidefinite correlation matrix where `d` is the
dimension (``d ≥ 2``) and `k` is the number of factor loadings (``1 ≤ k < d``).

# Examples
```julia-repl
julia> cor_rand(Float64, 4, 2)
#>4×4 Array{Float64,2}:
 1.0        0.276386   0.572837   0.192875
 0.276386   1.0        0.493806  -0.352386
 0.572837   0.493806   1.0       -0.450259
 0.192875  -0.352386  -0.450259   1.0

julia> cor_rand(4, 1)
4×4 Array{Float64,2}:
1.0       -0.800513   0.541379  -0.650587
-0.800513   1.0       -0.656411   0.788824
0.541379  -0.656411   1.0       -0.533473
-0.650587   0.788824  -0.533473   1.0

julia> cor_rand(4)
4×4 Array{Float64,2}:
  1.0        0.81691   -0.27188    0.289011
  0.81691    1.0       -0.44968    0.190938
 -0.27188   -0.44968    1.0       -0.102597
  0.289011   0.190938  -0.102597   1.0
```
"""
function cor_rand(T::Type{<:AbstractFloat}, d::Int, k::Int=dim-1; ensure_posdef::Bool=false)
    @assert d ≥ 2
    @assert 1 ≤ k < d

    d == 1 && return ones(T, 1, 1)

    W  = randn(T, d, k)
    S  = W * W' + diagm(rand(T, d))
    S2 = diagm(1 ./ sqrt.(diag(S)))
    R = S2 * S * S2

    cor_constrain!(R)

    if ensure_posdef
        cor_fastPD!(R)
    end

    return R
end
cor_rand(d::Int, k::Int=d-1; kwargs...) = cor_rand(Float64, d, k; kwargs...)
cor_rand(d::Real, k::Real=d-1; kwargs...) = cor_rand(Float64, Int(d), Int(k); kwargs...)
function cor_rand(T::Type{<:AbstractFloat}, d::Real, k::Real=d-1; kwargs...)
    return cor_rand(T, Int(d), Int(k); kwargs...)
end
