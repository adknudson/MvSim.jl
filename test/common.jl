using Test
using Bigsimr
using Distributions
using LinearAlgebra: tril, diag, issymmetric

cortypes = (Pearson, Spearman, Kendall)
types = (Float32, Float64)

@testset "is valid correlation" begin
    r_negdef = [
        1.00 0.82 0.56 0.44
        0.82 1.00 0.28 0.85
        0.56 0.28 1.00 0.22
        0.44 0.85 0.22 1.00
    ]
    @test !iscorrelation(r_negdef)

    r = cor_rand(10)
    @test iscorrelation(r)

    r[1] = nextfloat(1.0)
    @test !iscorrelation(r)
end

@testset "normal functions" begin
    @test Bigsimr._normpdf(-Inf) == 0.0
    @test Bigsimr._normpdf(0.0) == pdf(Normal(), 0.0)
    @test Bigsimr._normpdf(Inf) == 0.0

    @test Bigsimr._normcdf(-Inf) == 0.0
    @test Bigsimr._normcdf(0.0) == 0.5
    @test Bigsimr._normcdf(Inf) == 1.0

    @test Bigsimr._norminvcdf(0.0) == -Inf
    @test Bigsimr._norminvcdf(0.5) == 0.0
    @test Bigsimr._norminvcdf(1.0) == Inf
end

@testset "clamp to correlation" begin
    @test Bigsimr._cor_clamp(-2.0) == -1.0
    @test Bigsimr._cor_clamp( 0.0) ==  0.0
    @test Bigsimr._cor_clamp( 2.0) ==  1.0

    for T in types
        d = 1000
        x = rand(T, d) * 4 .- 2
        y = copy(x)

        Bigsimr._cor_clamp!(y)
        z = Bigsimr._cor_clamp(x)

        @test eltype(y) === eltype(z) === T
        @test all(-one(T) .≤ y .≤ one(T))
        @test all(-one(T) .≤ z .≤ one(T))
        @test y == z

        X = rand(T, d, d) * 4 .- 2
        Y = copy(X)

        l, u = T(-0.7), T(0.9)
        L = fill(l, d, d)
        U = fill(u, d, d)
        Bigsimr._cor_clamp!(Y, L, U)
        Z = Bigsimr._cor_clamp(X, L, U)

        @test eltype(Y) === eltype(Z) === T

        @test all(l .≤ tril(Y, -1) .≤ u)
        @test all(l .≤ tril(Z, -1) .≤ u)
        @test Y == Z
        @test diag(X) == diag(Y) == diag(Z)

        @test_throws AssertionError Bigsimr._cor_clamp(X, U, L)

        L2 = fill(l, d-1, d-1)
        @test_throws DimensionMismatch Bigsimr._cor_clamp(X, U, L2)

        Bigsimr._cor_clamp!(Y, L, U, set_diag=true)
        @test all(diag(Y) .== 1)
        Bigsimr._cor_clamp!(Y, L, U, ensure_symmetry=true)
        @test issymmetric(Y)
    end
end

@testset "constrain to correlation matrix" begin
    for T in types
        r = cor_rand(T, 10) * T(1.01)
        p = copy(r)

        @test !iscorrelation(r)

        q = Bigsimr._cor_constrain(r)
        Bigsimr._cor_constrain!(p)

        @test p == q
        @test eltype(p) === eltype(q) === T
        @test iscorrelation(p)
    end
end

@testset "covariance to correlation" begin
    S = [
        0.02500 0.00750 0.00175
        0.00750 0.00700 0.00135
        0.00175 0.00135 0.00043
    ]
    for T in types
        R = Matrix{T}(S)
        @test !iscorrelation(R)
        Bigsimr._cov2cor!(R)
        @test iscorrelation(R)
        @test eltype(R) === T
    end
end

@testset "other utilities" begin
    # set diag
    for T in types
        d = 10
        x = rand(T, d, d)
        y = T(0.5)
        Bigsimr._set_diag!(x, y)
        @test all(diag(x) .== y)
        @test eltype(x) === T
    end

    # symmetrize
    for T in types
        d = 10
        x = rand(T, d, d)
        Bigsimr._symmetrize!(x)
        @test issymmetric(x)
        @test eltype(x) === T
    end

end
