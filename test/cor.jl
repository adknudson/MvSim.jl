using Test
using Bigsimr

cortypes = (Pearson, Spearman, Kendall)
types = (Float32, Float64)

d = 10
n = 100

for T in types, C in cortypes
    x = randn(T, n)
    y = randn(T, n)
    z = randn(T, n, d)

    @test_nowarn cor(x, y, C()) # vector-vector
    @test_nowarn cor(x, z, C()) # matrix-vector
    @test_nowarn cor(z, x, C()) # vector-matrix
    @test_nowarn cor(z,    C()) # matrix
    @test_nowarn cor(z, z, C()) # matrix-matrix

    # TODO: Threaded not yet implemented
    @test_skip cor_threaded(x, y, C()) # vector-vector
    @test_skip cor_threaded(x, z, C()) # matrix-vector
    @test_skip cor_threaded(z, x, C()) # vector-matrix
    @test_skip cor_threaded(z,    C()) # matrix
    @test_skip cor_threaded(z, z, C()) # matrix-matrix

    # TODO: Wait for StatsBase to accept PR#673
    @test size(cor(x, y, C())) == ()     # vector-vector
    @test_skip size(cor(x, z, C())) == (1, d) # matrix-vector
    @test_skip size(cor(z, x, C())) == (d, 1) # vector-matrix
    @test size(cor(z,    C())) == (d, d) # matrix
    @test size(cor(z, z, C())) == (d, d) # matrix-matrix

    # TODO: fix type stability for Spearman/Kendall correlation
    @test_skip eltype(cor(x, y, C())) === T # vector-vector
    @test_skip eltype(cor(x, z, C())) === T # matrix-vector
    @test_skip eltype(cor(z, x, C())) === T # vector-matrix
    @test_skip eltype(cor(z,    C())) === T # matrix
    @test_skip eltype(cor(z, z, C())) === T # matrix-matrix
end
