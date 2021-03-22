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

@testset "pearson exact bounds" begin
    dists = (NegativeBinomial(20, 0.2), LogNormal(3, 1))
    # Must work for any univariate distribution
    for D1 in dists, D2 in dists
        @test_nowarn cor_bounds(D1, D2, Pearson(); use_exact=true)
        @test_nowarn cor_bounds(D1, D2, Pearson(); use_exact=true, hermite_degree=11)
    end
end
