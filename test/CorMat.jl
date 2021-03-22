using Test
using Bigsimr


@testset "constructors" begin
    r_negdef = [
        1.00 0.82 0.56 0.44
        0.82 1.00 0.28 0.85
        0.56 0.28 1.00 0.22
        0.44 0.85 0.22 1.00
    ]
    a = cor_rand(10)
    cortypes = (Pearson(), Spearman(), Kendall(), Adjusted(), nothing)
    for C in cortypes
        @test_nowarn CorMat(a, C)
        @test_nowarn CorMat{typeof(C)}(a)
        @test_nowarn CorMat(r_negdef, C)
        @test iscorrelation(CorMat(r_negdef, C))
    end
end

@testset "converting" begin
    a = cor_rand(10)
    # self to self returns self
    for C in (Pearson(), Spearman(), Kendall(), Adjusted(), nothing)
        m = CorMat(a, C)
        @test_nowarn convert(CorMat{typeof(C)}, m)
        @test typeof(convert(CorMat{typeof(C)}, m)) === typeof(m)
    end

    # Converting a matrix
    for C in (Pearson, Spearman, Kendall, Adjusted, Nothing)
        @test_nowarn convert(CorMat{C}, a)
        @test typeof(convert(CorMat{C}, a)) === CorMat{C}
    end

    # Pearson -> C
    m = CorMat{Pearson}(a)
    for C in (Spearman, Kendall, Nothing)
        @test_nowarn convert(CorMat{C}, m)
        @test typeof(convert(CorMat{C}, m)) === CorMat{C}
    end
    @test_throws MethodError convert(CorMat{Adjusted}, m)

    # Spearman -> C
    m = CorMat{Spearman}(a)
    for C in (Pearson, Kendall, Adjusted, Nothing)
        @test_nowarn convert(CorMat{C}, m)
        @test typeof(convert(CorMat{C}, m)) === CorMat{C}
    end

    # Kendall -> C
    m = CorMat{Kendall}(a)
    for C in (Pearson, Spearman, Adjusted, Nothing)
        @test_nowarn convert(CorMat{C}, m)
        @test typeof(convert(CorMat{C}, m)) === CorMat{C}
    end

    # Adjusted -> C
    m = CorMat{Adjusted}(a)
    for C in (Pearson, Spearman, Kendall)
        @test_throws MethodError convert(CorMat{C}, m)
    end
    @test_nowarn convert(CorMat{Nothing}, m)
    @test typeof(convert(CorMat{Nothing}, m)) === CorMat{Nothing}

    # Nothing -> C
    m = CorMat{Nothing}(a)
    for C in (Pearson, Spearman, Kendall, Adjusted)
        @test_nowarn convert(CorMat{C}, m)
        @test typeof(convert(CorMat{C}, m)) === CorMat{C}
    end
end

@testset "methods for CorMat" begin
    # getting the cortype
    for C in (Pearson, Spearman, Kendall, Adjusted, Nothing)
        a = cor_rand(10)
        m = CorMat{C}(a)

        @test cortype(m) === C
        @test cortype(CorMat{C}) === C
    end

    # turning random normal to correlated random normal
    for C in (Pearson, Spearman, Kendall, Adjusted, Nothing)
        d = 10
        n = 100
        a = cor_rand(d)
        m = CorMat{C}(a)
        x = Bigsimr._randn(n, d)
        r = similar(x)

        Bigsimr.unwhiten!(r, m, x)
        @test Bigsimr.unwhiten(m, x) == r

        z = Bigsimr.whiten(m, r)
        Bigsimr.whiten!(m, r)
        @test z == r
    end

    # generate random multivariate normal given a CorMat
    d = 10
    n = 100
    a = cor_rand(d)
    m = CorMat{Pearson}(a)
    @test_nowarn Bigsimr._rmvn(n, m)
    m = CorMat{Spearman}(a)
    @test_throws MethodError Bigsimr._rmvn(n, m)
    m = CorMat{Kendall}(a)
    @test_throws MethodError Bigsimr._rmvn(n, m)
    m = CorMat{Adjusted}(a)
    @test_nowarn Bigsimr._rmvn(n, m)
    m = CorMat{Nothing}(a)
    @test_nowarn Bigsimr._rmvn(n, m)
end
