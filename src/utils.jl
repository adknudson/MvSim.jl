# Check if a matrix is a valid correlation matrix
function iscorrelation(X::Matrix{T}) where {T<:Real}
    all([
        isposdef(X),
        issymmetric(X),
        all(diag(X) .== one(T)),
        all(-one(T) .≤ X .≤ one(T))
    ])
end


for T in (Float64, Float32)
    @eval _normpdf(x::$T) = exp(-abs2(x)/2) * $T(invsqrt2π)
    @eval _normcdf(x::$T) = erfc(-x * $T(invsqrt2)) / 2
    @eval _norminvcdf(x::$T) = -$T(sqrt2) * erfcinv(2x)
end
for F in (:_normpdf, :_normcdf, :_norminvcdf)
    @eval $F(x::Float16) = Float16($F(Float32(x)))
end
for T in (Int16, Int32, Int64, Real)
    for F in (:_normpdf, :_normcdf, :_norminvcdf)
        @eval $F(x::$T) = $F(Float64(x))
    end
end


# convert correlation types: _from_to(x)
_pe_sp(x) = asin(x / 2) * 6 / π
_pe_ke(x) = asin(x) * 2 / π
_sp_pe(x) = sin(x * π / 6) * 2
_sp_ke(x) = asin(sin(x * π / 6) * 2) * 2 / π
_ke_pe(x) = sin(x * π / 2)
_ke_sp(x) = asin(sin(x * π / 2) / 2) * 6 / π
