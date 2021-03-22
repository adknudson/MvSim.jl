using Test
using Bigsimr
using Distributions
using LinearAlgebra: tril, diag, issymmetric
using Polynomials

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

@testset "correlation conversion functions" begin
    fun = (
        Bigsimr._pe_sp,
        Bigsimr._pe_ke,
        Bigsimr._sp_pe,
        Bigsimr._sp_ke,
        Bigsimr._ke_pe,
        Bigsimr._ke_sp
    )
    for T in types, f in fun
        x = T[-1, 0, 1]
        @test f.(x) ≈ x
        @test typeof(f(x[1])) == T
    end
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
        y = rand(T)
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

@testset "Core Hermite Function" begin
    # Must work for any real input
    types = (Float64, Float32, Float16, BigFloat, Int128, Int64, Int32, Int16, BigInt, Rational)
    for T in types
        @test_nowarn Bigsimr._hermite(one(T), 5)
    end

    # For the following types, the input type should be the same as the output
    types = (Float64, Float32, Float16, BigFloat, Int64, BigInt, Rational)
    for T in types
        @test Bigsimr._hermite(one(T), 5) isa T
    end

    @test_nowarn Bigsimr._hermite(3.14, 5.0)
    @test_throws InexactError Bigsimr._hermite(3.14, 5.5)

    # Must work for arrays/matrices/vectors
    A = rand(3)
    B = rand(3, 3)
    C = rand(3, 3, 3)
    @test_nowarn Bigsimr._hermite(A, 3)
    @test_nowarn Bigsimr._hermite(B, 3)
    @test_nowarn Bigsimr._hermite(C, 3)
end

@testset "Hermite-Normal PDF" begin
    @test iszero(Bigsimr._Hp(Inf, 10))
    @test iszero(Bigsimr._Hp(-Inf, 10))
    @test 1.45182435 ≈ Bigsimr._Hp(1.0, 5)
end

@testset "Get Hermite Coefficients" begin
    dists = (
        Binomial(20, 0.2),
        NegativeBinomial(20, 0.002),
        LogitNormal(3, 1),
        Beta(5, 3)
    )

    for D in dists
        @test_nowarn Bigsimr._get_hermite_coefs(D, 7)
        @test_nowarn Bigsimr._get_hermite_coefs(D, 7.0)
        @test_throws InexactError Bigsimr._get_hermite_coefs(D, 7.5)
    end
end

@testset "Solve Polynomial on [-1, 1]" begin
    r1 = -1.0
    r2 = 1.0
    r3 = eps()
    r4 = 2 * rand() - 1

    P1 = coeffs(3 * fromroots([r1, 7, 7, 8]))
    P2 = coeffs(-5 * fromroots([r2, -1.14, -1.14, -1.14, -1.14, 1119]))
    P3 = coeffs(1.2 * fromroots([r3, nextfloat(1.0), prevfloat(-1.0)]))
    P4 = coeffs(fromroots([-5, 5, r4]))
    P5 = coeffs(fromroots([nextfloat(1.0), prevfloat(-1.0)]))
    P6 = coeffs(fromroots([-0.5, 0.5]))

    # One root at -1.0
    @test Bigsimr._solve_poly_pm_one(P1) ≈ r1 atol=0.001
    # One root at 1.0
    @test Bigsimr._solve_poly_pm_one(P2) ≈ r2 atol=0.001
    # Roots that are just outside [-1, 1]
    @test Bigsimr._solve_poly_pm_one(P3) ≈ r3 atol=0.001
    @test Bigsimr._solve_poly_pm_one(P4) ≈ r4 atol=0.001
    # Case of no roots
    @test isnan(Bigsimr._solve_poly_pm_one(P5))
    # Case of multiple roots
    @test length(Bigsimr._solve_poly_pm_one(P6)) == 2
    @test Bigsimr._nearest_root(-0.6, Bigsimr._solve_poly_pm_one(P6)) ≈ -0.5 atol=0.001
end

@testset "random normal generation" begin
    for T in (Float16, Float32, Float64)
        d = 10
        n = 100

        @test_nowarn Bigsimr._randn(T, d, d)
        @test_nowarn Bigsimr._randn(T, T(d), T(d))
        @test_throws InexactError Bigsimr._randn(T, 10.5, 11)

        x = Bigsimr._randn(T, 100, 10)
        @test eltype(x) === T

        r = cor_rand(T, d)
        @test_nowarn Bigsimr._rmvn(n, r)
        @test size(Bigsimr._rmvn(n, r)) == (n,d)
        @test size(Bigsimr._rmvn(n, 0.5)) == (n,2)
    end
end
