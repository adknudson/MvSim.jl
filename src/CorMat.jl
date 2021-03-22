struct CorMat{C<:CorOrNothing} <: AbstractMatrix{Float64}
    mat::AbstractMatrix{Float64}
    chol::Cholesky{Float64, <:AbstractMatrix{Float64}}
    cortype::C
end
function CorMat(m::Matrix{Float64}, C::CorOrNothing)
    if !iscorrelation(m)
        m = cor_near_posdef(_cor_constrain(m))
    end

    chol = cholesky(m)

    return CorMat(m, chol, C)
end
CorMat(m::Matrix{Float64}) = CorMat(m, nothing)
CorMat{Nothing}(m::Matrix{Float64}) = CorMat(m, nothing)
CorMat{C}(m::Matrix{Float64}) where {C<:AbstractCorrelation} = CorMat(m, C())


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
    rdiv!(r, m.chol.U)
    return r
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


# Pearson, Spearman, Kendall
Base.convert(::Type{CorMat{Spearman}}, m::CorMat{Pearson }) = CorMat(_pe_sp.(m.mat), Spearman())
Base.convert(::Type{CorMat{Kendall }}, m::CorMat{Pearson }) = CorMat(_pe_ke.(m.mat), Kendall())
Base.convert(::Type{CorMat{Pearson }}, m::CorMat{Spearman}) = CorMat(_sp_pe.(m.mat), Pearson())
Base.convert(::Type{CorMat{Kendall }}, m::CorMat{Spearman}) = CorMat(_pe_ke.(m.mat), Kendall())
Base.convert(::Type{CorMat{Pearson }}, m::CorMat{Kendall }) = CorMat(_ke_pe.(m.mat), Pearson())
Base.convert(::Type{CorMat{Spearman}}, m::CorMat{Kendall }) = CorMat(_ke_sp.(m.mat), Spearman())
# Converting to the same returns the same
Base.convert(::Type{CorMat{C}}, m::CorMat{C}) where {C<:AbstractCorrelation} = m
# Nothing Type (just change cortype to nothing or to the target)
Base.convert(::Type{CorMat{Nothing}}, m::CorMat{<:CorOrNothing}) = CorMat(m.mat, m.chol, nothing)
Base.convert(::Type{CorMat{C}}, m::CorMat{Nothing}) where {C<:AbstractCorrelation} = CorMat(m.mat, m.chol, C())
# Adjusted Type (cannot convert Pearson to adjusted without margins)
Base.convert(::Type{CorMat{Adjusted}}, m::CorMat{Spearman}) = CorMat(_sp_pe.(m.mat), Adjusted())
Base.convert(::Type{CorMat{Adjusted}}, m::CorMat{Kendall }) = CorMat(_ke_pe.(m.mat), Adjusted())
# Converting a plain matrix is basically just a constructor
Base.convert(::Type{CorMat{C}}, m::AbstractMatrix{Float64}) where {C<:CorOrNothing} = CorMat(m, C())

# Alternative form
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


iscorrelation(C::CorMat) = iscorrelation(C.mat)


_rmvn(n::Int, m::CorMat{Pearson}) = unwhiten(m, _randn(n, size(m,1)))
_rmvn(n::Int, m::CorMat{Adjusted}) = unwhiten(m, _randn(n, size(m,1)))
_rmvn(n::Int, m::CorMat{Nothing}) = unwhiten(m, _randn(n, size(m,1)))
