using LinearAlgebra
using Bigsimr: iscorrelation, cor_nearPD

abstract type AbstractCorrelation end
struct Pearson  <: AbstractCorrelation end
struct Spearman <: AbstractCorrelation end
struct Kendall  <: AbstractCorrelation end

struct CorMat{T<:Real, C<:Union{AbstractCorrelation, Nothing}} <: AbstractMatrix{T}
    mat::AbstractMatrix
    chol::Cholesky{T, <:AbstractMatrix}
    cor_type::C
end
function CorMat(m::AbstractMatrix, C::Union{AbstractCorrelation, Nothing})
    if !iscorrelation(m)
        m = cor_nearPD(m)
    end

    chol = cholesky(m)

    return CorMat(m, chol, C)
end
CorMat(m::AbstractMatrix) = CorMat(m, nothing)

cor_type(m::CorMat) = m.cor_type

Base.size(m::CorMat) = size(m.mat)
Base.getindex(m::CorMat, i::Int) = getindex(m.mat, i)
Base.getindex(m::CorMat, I::Vararg{Int, N}) where {N} = getindex(m.mat, I...)
Base.setindex!(m::CorMat, v, i::Int) = setindex!(m.mat, v, i)
Base.setindex!(m::CorMat, v, I::Vararg{Int, N}) where {N} = setindex!(m.mat, v, I...)

_pe_sp(x) = asin(x / 2) * 6 / π
_pe_ke(x) = asin(x) * 2 / π
_sp_pe(x) = sin(x * π / 6) * 2
_sp_ke(x) = asin(sin(x * π / 6) * 2) * 2 / π
_ke_pe(x) = sin(x * π / 2)
_ke_sp(x) = asin(sin(x * π / 2) / 2) * 6 / π

pairs = (
    (Pearson,  Spearman, _pe_sp),
    (Pearson,  Kendall,  _pe_ke),
    (Spearman, Pearson,  _sp_pe),
    (Spearman, Kendall,  _pe_ke),
    (Kendall,  Pearson,  _ke_pe),
    (Kendall,  Spearman, _ke_sp)
)
for s in pairs
    # from, to, fun
    C, D, fun = s
    ts = (Float64, Float32)
    for T in ts, S in ts
        @eval function Base.convert(::Type{CorMat{$T, $D}}, m::CorMat{$S, $C})
            return CorMat($T.($fun.(m.mat)), $D())
        end
    end
end
Base.convert(::Type{CorMat{T,C}}, m::CorMat{T,C}) where {T,C} = m
