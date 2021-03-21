# Check if a matrix is a valid correlation matrix
function iscorrelation(X::Matrix{T}) where {T<:Real}
    all([
        isposdef(X),
        issymmetric(X),
        all(diag(X) .== one(T)),
        all(-one(T) .≤ X .≤ one(T))
    ])
end


# normal distribution functions
_normpdf(x::Real) = exp(-abs2(x)/2) * invsqrt2π
_normcdf(x::Real) = erfc(-x * invsqrt2) / 2
_norminvcdf(x::Real) = -sqrt2 * erfcinv(2x)


# convert correlation types: _from_to(x)
_pe_sp(x) = asin(x / 2) * 6 / π
_pe_ke(x) = asin(x) * 2 / π
_sp_pe(x) = sin(x * π / 6) * 2
_sp_ke(x) = asin(sin(x * π / 6) * 2) * 2 / π
_ke_pe(x) = sin(x * π / 2)
_ke_sp(x) = asin(sin(x * π / 2) / 2) * 6 / π


# set diagonal of a matrix
function _set_diag!(X::Matrix{T}, y::T) where {T}
    X[diagind(X)] .= y
    return X
end
_set_diag(X, y) = _set_diag!(copy(X), y)


# clamp to correlation
_cor_clamp!(r::AbstractArray) = clamp!(r, -one(eltype(r)), one(eltype(r)))
_cor_clamp(r::AbstractArray) = _cor_clamp!(copy(r))
_cor_clamp(r::Real) = clamp(r, -one(r), one(r))
function _cor_clamp!(
    C::Matrix{<:Real}, lo::Matrix{<:Real}, hi::Matrix{<:Real};
    set_diag::Bool=false, ensure_symmetry::Bool=false
)
    !(size(C) == size(lo) == size(hi)) && throw(DimensionMismatch("The dimensions of the matrix and the bounds must match."))

    L = copy(lo)
    U = copy(hi)

    L[diagind(L)] .= -Inf
    U[diagind(U)] .=  Inf

    !all(L .≤ U) && throw(AssertionError("All lower bounds must be less than all upper bounds."))

    C .= clamp.(C, L, U)

    set_diag && _set_diag!(C, one(eltype(C)))
    ensure_symmetry && _symmetrize!(C)

    return C
end
function _cor_clamp(
    C::Matrix{<:Real}, lo::Matrix{<:Real}, hi::Matrix{<:Real};
    set_diag::Bool=false, ensure_symmetry::Bool=false
)
    return _cor_clamp!(copy(C), lo, hi; set_diag=set_diag, ensure_symmetry=ensure_symmetry)
end


# make a matrix symmetric
function _symmetrize!(X::Matrix{<:Real}, uplo=:U)
    X .= Symmetric(X, uplo)
    return X
end
_symmetrize(X::Matrix{<:Real}, uplo=:U) = _symmetrize!(copy(X), uplo)


# constrain a correlation matrix to have diag=1 and all off diagonal values in [-1, 1]
function _cor_constrain!(C::Matrix{<:Real}, uplo=:U)
    _cor_clamp!(C)
    _symmetrize!(C, uplo)
    C[diagind(C)] .= one(eltype(C))
    return C
end
_cor_constrain(C::Matrix{<:Real}, uplo=:U) = _cor_constrain!(copy(C), uplo)


# project a covariance matrix onto a correlation matrix
function _cov2cor!(C::Matrix{<:AbstractFloat})
    D = sqrt(inv(Diagonal(C)))
    C .= D * C * D
    _cor_constrain!(C)
    return C
end
_cov2cor(C::Matrix{<:AbstractFloat}) = _cov2cor!(copy(C))


_normal_to_margin(dist::UD, x::Float64) = quantile(dist, _normcdf(x))
_normal_to_margin(dist::UD, x::Real) = _normal_to_margin(dist, Float64(x))
_normal_to_margin(dist::UD, A::AbstractArray) = _normal_to_margin.(dist, A)


# hermite polynomial evaluation
function _hermite(x::T, n::Int) where {T<:Real}
    if n == 0
        return one(T)
    elseif n == 1
        return x
    end

    Hkp1, Hk, Hkm1 = zero(T), x, one(T)
    for k in 2:n
        Hkp1 = x*Hk - (k-1) * Hkm1
        Hkm1, Hk = Hk, Hkp1
    end
    Hkp1
end
_hermite(x::Real, n::Real) = _hermite(x, Int(n))
_hermite(A::AbstractArray, n::Real) = _hermite.(A, Ref(n))


# We need to account for when x is ±∞ otherwise Julia will return NaN for 0×∞
function _Hp(x::Real, n::Int)
    isinf(x) ? zero(x) : _hermite(x, n) * _normpdf(x)
end
_Hp(x::Real, n::Real) = _Hp(x, Int(n))


function _get_hermite_coefs(dist::UD, n::Int)
    ak = zeros(Float64, n + 1)
    m = 2n
    ts, ws = gausshermite(m)
    ts    .= ts * sqrt2
    xs = _normal_to_margin(dist, ts)

    ak = [sum(ws .* _hermite(ts, k) .* xs) for k in 0:n]

    return invsqrtpi * ak ./ factorial.(0:n)
end
_get_hermite_coefs(dist::UD, n::Real) = _get_hermite_coefs(dist, Int(n))


function _Gn0d(n::Int, A::UnitRange{Int}, B::UnitRange{Int}, α::Vector{Float64}, β::Vector{Float64}, σAσB_inv::Float64)
    if n == 0
        return 0.0
    end
    M = length(A)
    N = length(B)
    accu = 0.0
    for r=1:M, s=1:N
        r11 = _Hp(α[r+1], n-1) * _Hp(β[s+1], n-1)
        r00 = _Hp(α[r],   n-1) * _Hp(β[s],   n-1)
        r01 = _Hp(α[r],   n-1) * _Hp(β[s+1], n-1)
        r10 = _Hp(α[r+1], n-1) * _Hp(β[s],   n-1)
        accu += A[r]*B[s] * (r11 + r00 - r01 - r10)
    end
    accu * σAσB_inv
end

function _Gn0m(n::Int, A::UnitRange{Int}, α::Vector{Float64}, dB::UD, σAσB_inv::Float64)
    if n == 0
        return 0.0
    end
    M = length(A)
    accu = 0.0
    for r=1:M
        accu += A[r] * (_Hp(α[r+1], n-1) - _Hp(α[r], n-1))
    end
    m = n + 4
    t, w = gausshermite(m)
    t .= t * sqrt2
    X = _normal_to_margin(dB, t)
    S = invsqrtpi * sum(w .* _hermite(t, n) .* X)
    return -σAσB_inv * accu * S
end


function _solve_poly_pm_one(coef::Vector{<:Real})
    P = Polynomial(coef)
	dP = derivative(P)
    r = roots(x->P(x), x->dP(x), interval(-1, 1), Krawczyk, 1e-3)

    length(r) == 1 && return mid(r[1].interval)
    length(r) == 0 && return NaN
    length(r) >  1 && return [mid(rs.interval) for rs in r]
end
_nearest_root(target::Real, roots::Vector{<:Real}) = roots[argmin(abs.(roots .- target))]


# internal threaded random normal sampler
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


# internal random multivariate normal
for T in (Float64, Float32, Float16)
    @eval function _rmvn(n::Int, ρ::Matrix{$T})
        Z = _randn($T, n, size(ρ, 1))
        C = cholesky(ρ)
        Z * Matrix{$T}(C.U)
    end
end
_rmvn(n::Int, ρ::Float64) = _rmvn(n, [1.0 ρ; ρ 1.0])
