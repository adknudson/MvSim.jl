using Test
using Bigsimr

cortypes = (Pearson, Spearman, Kendall)
types = (Float32, Float64)

# Converting type A -> A must result in the same matrix
for C in cortypes
    r = cor_rand(4)
    p = copy(r)
    cor_convert!(p, C(), C())
    @test r == cor_convert(r, C(), C())

end

# Must map (-1, 0, 1) onto itself within numerical error
for C1 in cortypes, C2 in cortypes
    @test cor_convert( 0.0, C1(), C2())  == 0.0
    @test cor_convert( 1.0, C1(), C2())   ≤ 1.0
    @test cor_convert( 1.0, C1(), C2())   ≈ 1.0
    @test cor_convert(-1.0, C1(), C2())  ≥ -1.0
    @test cor_convert(-1.0, C1(), C2())  ≈ -1.0

    r = cor_rand(4)
    p = copy(r)
    cor_convert!(p, C1(), C2())
    @test cor_convert(r, C1(), C2()) == p
end

# Must work for each type
# Must respect these input eltypes
for T in types, C1 in cortypes, C2 in cortypes
    @test_nowarn cor_convert(T(0.5), C1(), C2())
    @test eltype(cor_convert(T(0.5), C1(), C2())) === T
end
