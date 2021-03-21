using Test
using Bigsimr

const tests = [
    "common",
    "cor_adjust",
    "cor_bounds",
    "cor_convert",
    "cor_fast_posdef",
    "cor_near_posdef",
    "cor_rand",
    "cor",
    "CorMat",
    "MvDist",
    "GSDist",
    "randvec"
]

printstyled("Running tests:\n", color=:blue)

for t in tests
    @testset "Test $t" begin
        include("$t.jl")
    end
end
