using Test
using Bigsimr
using Polynomials
using Distributions

@testset "Pearson Correlation Matching" begin
    dA = Beta(2, 3)
    dB = Binomial(2, 0.2)
    dC = Binomial(20, 0.2)

    @testset "Continuous-Continuous" begin
        @test -0.914 ≈ pearson_match(-0.9, dA, dA) atol=0.05
        @test -0.611 ≈ pearson_match(-0.6, dA, dA) atol=0.05
        @test -0.306 ≈ pearson_match(-0.3, dA, dA) atol=0.05
        @test  0.304 ≈ pearson_match( 0.3, dA, dA) atol=0.05
        @test  0.606 ≈ pearson_match( 0.6, dA, dA) atol=0.05
        @test  0.904 ≈ pearson_match( 0.9, dA, dA) atol=0.05
    end

    @testset "Discrete-Discrete" begin
        @test -0.937 ≈ pearson_match(-0.5, dB, dB) atol=0.05
        @test -0.501 ≈ pearson_match(-0.3, dB, dB) atol=0.05
        @test -0.322 ≈ pearson_match(-0.2, dB, dB) atol=0.05
        @test  0.418 ≈ pearson_match( 0.3, dB, dB) atol=0.05
        @test  0.769 ≈ pearson_match( 0.6, dB, dB) atol=0.05
        @test  0.944 ≈ pearson_match( 0.8, dB, dB) atol=0.05

        @test -0.939 ≈ pearson_match(-0.9, dC, dC) atol=0.05
        @test -0.624 ≈ pearson_match(-0.6, dC, dC) atol=0.05
        @test -0.311 ≈ pearson_match(-0.3, dC, dC) atol=0.05
        @test  0.310 ≈ pearson_match( 0.3, dC, dC) atol=0.05
        @test  0.618 ≈ pearson_match( 0.6, dC, dC) atol=0.05
        @test  0.925 ≈ pearson_match( 0.9, dC, dC) atol=0.05
    end

    @testset "Mixed" begin
        @test -0.890 ≈ pearson_match(-0.7, dB, dA) atol=0.05
        @test -0.632 ≈ pearson_match(-0.5, dB, dA) atol=0.05
        @test -0.377 ≈ pearson_match(-0.3, dB, dA) atol=0.05
        @test  0.366 ≈ pearson_match( 0.3, dB, dA) atol=0.05
        @test  0.603 ≈ pearson_match( 0.5, dB, dA) atol=0.05
        @test  0.945 ≈ pearson_match( 0.8, dB, dA) atol=0.05

        @test -0.928 ≈ pearson_match(-0.9, dC, dA) atol=0.05
        @test -0.618 ≈ pearson_match(-0.6, dC, dA) atol=0.05
        @test -0.309 ≈ pearson_match(-0.3, dC, dA) atol=0.05
        @test  0.308 ≈ pearson_match( 0.3, dC, dA) atol=0.05
        @test  0.613 ≈ pearson_match( 0.6, dC, dA) atol=0.05
        @test  0.916 ≈ pearson_match( 0.9, dC, dA) atol=0.05
    end

end
