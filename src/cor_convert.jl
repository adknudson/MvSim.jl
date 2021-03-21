"""
    cor_convert(X::Matrix{<:Real}, from::Union{Pearson, Spearman, Kendall}, to::Union{Pearson, Spearman, Kendall})

Convert from one type of correlation matrix to another.

The role of conversion in this package is typically used from either Spearman or
Kendall to Pearson where the Pearson correlation is used in the generation of
random multivariate normal samples. After converting, the correlation matrix
may not be positive semidefinite, so it is recommended to check using
`LinearAlgebra.isposdef`, and subsequently calling [`cor_nearPD`](@ref).

See also: [`cor_nearPD`](@ref), [`cor_fastPD`](@ref)

The possible correlation types are:
* `Pearson`
* `Spearman`
* `Kendall`

# Examples
```jldoctest
julia> r = [ 1.0       -0.634114   0.551645   0.548993
            -0.634114   1.0       -0.332105  -0.772114
             0.551645  -0.332105   1.0        0.143949
             0.548993  -0.772114   0.143949   1.0];

julia> cor_convert(r, Pearson(), Spearman())
4×4 Array{Float64,2}:
  1.0       -0.616168   0.533701   0.531067
 -0.616168   1.0       -0.318613  -0.756979
  0.533701  -0.318613   1.0        0.13758
  0.531067  -0.756979   0.13758    1.0

julia> cor_convert(r, Spearman(), Kendall())
4×4 Array{Float64,2}:
  1.0       -0.452063   0.385867    0.383807
 -0.452063   1.0       -0.224941   -0.576435
  0.385867  -0.224941   1.0         0.0962413
  0.383807  -0.576435   0.0962413   1.0

julia> r == cor_convert(r, Pearson(), Pearson())
true
```
"""
function cor_convert end
cor_convert(x::Real, from::C,        to::C) where {C<:AbstractCorrelation} = x
cor_convert(x::Real, from::Pearson,  to::Spearman) = _cor_clamp(_pe_sp(x))
cor_convert(x::Real, from::Pearson,  to::Kendall)  = _cor_clamp(_pe_ke(x))
cor_convert(x::Real, from::Spearman, to::Pearson)  = _cor_clamp(_sp_pe(x))
cor_convert(x::Real, from::Spearman, to::Kendall)  = _cor_clamp(_sp_ke(x))
cor_convert(x::Real, from::Kendall,  to::Pearson)  = _cor_clamp(_ke_pe(x))
cor_convert(x::Real, from::Kendall,  to::Spearman) = _cor_clamp(_ke_sp(x))
function cor_convert!(C::Matrix{<:Real}, from::PeSpKe, to::PeSpKe)
    C .= cor_convert.(C, Ref(from), Ref(to))
    _cor_constrain!(C)
    return C
end
cor_convert(C::Matrix{<:Real}, from::PeSpKe, to::PeSpKe) = cor_convert!(copy(C), from, to)
