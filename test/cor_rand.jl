using Test
using Bigsimr

types = (Float16, Float32, Float64)

# The element type must be respected
d = 4
for T in types
    @test eltype(cor_rand(T, d)) === T
end

# Must work for numbers with integer representations
d = 4
for T in types
    @test_nowarn cor_rand(T(d))
    for S in types
        @test_nowarn cor_rand(S, T(d))
        for U in types
            @test_nowarn cor_rand(S, T(4), U(3))
        end
    end
end

# `d` must not be less than 2
for d in (-2, -1, 0, 1)
    @test_throws AssertionError cor_rand(d)
    for T in types
        @test_throws AssertionError cor_rand(T, d)
    end
end

# `k` must not be equal to or larger than `d`
d = 4
for k in (4, 5, 6)
    @test_throws AssertionError cor_rand(d, k)
    for T in types
        @test_throws AssertionError cor_rand(T, d, k)
    end
end


# `k` must not be less than 1
d = 4
for k in (-2, -1, 0)
    @test_throws AssertionError cor_rand(d, k)
    for T in types
        @test_throws AssertionError cor_rand(T, d, k)
    end
end
