using Test
using Bigsimr
using Distributions

cortypes = (Pearson, Spearman, Kendall)

@testset "stochastic bounds" begin
    dists = (NegativeBinomial(20, 0.2), LogNormal(3, 1))
    # Must work for any univariate distribution and correlation type
    for D1 in dists, D2 in dists, C in cortypes
        @test_nowarn cor_bounds(D1, D2, C(); use_exact=false)
        # Must work for any number with an integer representation
        types = (Float64, Float32, Float16, Int64, Int32, Int16)
        for T in types
            @test_nowarn cor_bounds(D1, D2, C(); n_samples=T(10_000), use_exact=false)
        end

        types = (Float64, Float32)
        for T in types
            @test_throws InexactError cor_bounds(
                D1, D2, C();
                n_samples=T(10.5), use_exact=false
            )
        end
    end
end


@testset "pearson exact method utilities" begin
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

    # Must work for any real input
    test_types = (Float64, Float32, Float16, BigFloat, Int128, Int64, Int32, Int16, BigInt, Rational)
    for T in test_types
        @test_nowarn Bigsimr._hermite(one(T), 5)
    end

    # For the following types, the input type should be the same as the output
    test_types = (Float64, Float32, Float16, BigFloat, Int64, BigInt, Rational)
    for T in test_types
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


@testset "pearson exact bounds" begin
    dists = (NegativeBinomial(20, 0.2), LogNormal(3, 1))
    # Must work for any univariate distribution
    for D1 in dists, D2 in dists
        @test_nowarn cor_bounds(D1, D2, Pearson(); use_exact=true)
        @test_nowarn cor_bounds(D1, D2, Pearson(); use_exact=true, hermite_degree=11)
    end
end
