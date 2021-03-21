using Test
using Bigsimr

r_negdef = [
    1.00 0.82 0.56 0.44
    0.82 1.00 0.28 0.85
    0.56 0.28 1.00 0.22
    0.44 0.85 0.22 1.00
]
q = copy(r_negdef)

@test !iscorrelation(r_negdef)

r = cor_near_posdef(r_negdef)
@test iscorrelation(r)

# Must respect input eltype
types = (Float64, Float32)
for T in types
    @test eltype(cor_near_posdef(Matrix{T}(r_negdef))) === T
end
