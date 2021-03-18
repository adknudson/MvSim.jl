using LinearAlgebra
using Bigsimr: iscorrelation, cor_nearPD


abstract type AbstractCorrelation end
struct Pearson  <: AbstractCorrelation end
struct Spearman <: AbstractCorrelation end
struct Kendall  <: AbstractCorrelation end


const CorOrNothing = Union{AbstractCorrelation, Nothing}


struct CorMat{C<:CorOrNothing} <: AbstractMatrix{Float64}
    mat::AbstractMatrix{Float64}
    chol::Cholesky{Float64, <:AbstractMatrix{Float64}}
    cortype::C
end

function CorMat(m::Matrix{Float64}, C::CorOrNothing)
    if !iscorrelation(m)
        m = cor_nearPD(m)
    end

    chol = cholesky(m)

    return CorMat(m, chol, C)
end
CorMat(m::Matrix{Float64}) = CorMat(m, nothing)
CorMat{Nothing}(m::Matrix{Float64}) = CorMat(m, nothing)
function CorMat{C}(m::Matrix{Float64}) where {C<:AbstractCorrelation}
    return CorMat(m, C())
end

cortype(::CorMat{C}) where {C<:CorOrNothing} = C
cortype(::Type{CorMat{C}}) where {C<:CorOrNothing} = C

function X_A_Xt(a::CorMat, x::Matrix{Float64})
    z = x * a.chol.L
    return z * transpose(z)
end
function Xt_A_X(a::CorMat, x::Matrix{Float64})
    z = transpose(x) * a.chol.L
    return z * transpose(z)
end
function X_invA_Xt(a::CorMat, x::Matrix{Float64})
    z = a.chol \ transpose(x)
    return x * z
end
function Xt_invA_X(a::CorMat, x::Matrix{Float64})
    z = a.chol \ x
    return transpose(x) * z
end

function whiten!(r::VecOrMat{Float64}, m::CorMat, x::VecOrMat{Float64})
    copyto!(r, x)
    return rdiv!(r, m.chol.U)
end
whiten!(m::CorMat, x::VecOrMat{Float64}) = whiten!(x, m, x)
whiten(m::CorMat, x::VecOrMat{Float64}) = whiten!(similar(x), m, x)
unwhiten!(r::VecOrMat{Float64}, m::CorMat, x::VecOrMat{Float64}) = mul!(r, x, m.chol.U)
unwhiten!(m::CorMat, x::VecOrMat{Float64}) = unwhiten!(x, m, x)
unwhiten(m::CorMat, x::VecOrMat{Float64}) = unwhiten!(similar(x), m, x)

Base.size(m::CorMat) = size(m.mat)
Base.getindex(m::CorMat, i::Int) = getindex(m.mat, i)
Base.getindex(m::CorMat, I::Vararg{Int, N}) where {N} = getindex(m.mat, I...)
Base.setindex!(m::CorMat, v, i::Int) = setindex!(m.mat, v, i)
Base.setindex!(m::CorMat, v, I::Vararg{Int, N}) where {N} = setindex!(m.mat, v, I...)

# from -> to
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
    # from, to, formula
    C, D, fun = s
    @eval function Base.convert(::Type{CorMat{$D}}, m::CorMat{$C})
        return CorMat($fun.(m.mat), $D())
    end
end
Base.convert(::Type{CorMat{C}}, m::CorMat{C}) where {C<:AbstractCorrelation} = m
function Base.convert(::Type{CorMat{Nothing}}, m::CorMat{<:CorOrNothing})
    return CorMat(m.mat, m.chol, nothing)
end
function Base.convert(::Type{CorMat{C}}, m::CorMat{Nothing}) where {C<:AbstractCorrelation}
    return CorMat(m.mat, m.chol, C())
end
CorMat{C}(m::CorMat) where {C<:CorOrNothing} = convert(CorMat{C}, m)

Base.Matrix(m::CorMat) = Matrix(m.mat)
Base.:*(m::CorMat, c::T) where {T<:Real} = m.mat * c
Base.:*(m::CorMat, x::VecOrMat{Float64}) = m.mat * x
Base.:\(m::CorMat, x::VecOrMat{Float64}) = m.chol \ x

LinearAlgebra.diag(m::CorMat) = ones(eltype(m), size(m,1))
LinearAlgebra.cholesky(m::CorMat) = m.chol
LinearAlgebra.inv(m::CorMat) = inv(m.chol)
LinearAlgebra.det(m::CorMat) = det(m.chol)
LinearAlgebra.logdet(m::CorMat) = logdet(m.chol)
