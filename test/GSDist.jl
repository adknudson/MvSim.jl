using Test
using Bigsimr
using Distributions

D = Gamma(2, 2)
G = Bigsimr.GSDist(D)

@test_nowarn params(G)

@test_nowarn mean(G)
@test_nowarn var(G)
@test_nowarn std(G)
@test_nowarn median(G)

x = median(G)
y = median(G) + 1

@test_nowarn quantile(G, 0.3)
@test_nowarn cquantile(G, 0.3)

@test_nowarn sampler(G)
@test_nowarn rand(G, 10)

# correlation adjustment
dA = NegativeBinomial(20, 0.1)
dB = Gamma(100, 4)

# Converting must retain accuracy
@test Bigsimr._pearson_match(-0.9, dA, dA) ≈ Bigsimr._pearson_match(-0.9, dA, dA; use_gsdist=false) atol=0.05
@test_skip Bigsimr._pearson_match(-0.9, dA, dB) ≈ Bigsimr._pearson_match(-0.9, dA, dB; use_gsdist=false) atol=0.05
@test_skip Bigsimr._pearson_match(-0.9, dB, dA) ≈ Bigsimr._pearson_match(-0.9, dB, dA; use_gsdist=false) atol=0.05
@test Bigsimr._pearson_match(-0.9, dB, dB) ≈ Bigsimr._pearson_match(-0.9, dB, dB; use_gsdist=false) atol=0.05
