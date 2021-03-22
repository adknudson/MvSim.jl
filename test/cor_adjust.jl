using Test
using Bigsimr
using Distributions

# Spearman/Kendall
d = 20
sp = cor_rand(d) |> m -> cor_convert(m, Pearson(), Spearman()) |> m -> CorMat{Spearman}(m)
ke = cor_rand(d) |> m -> cor_convert(m, Pearson(), Kendall()) |> m -> CorMat{Kendall}(m)

cor_adjust(sp)
cor_adjust(ke)

# Pearson
d = 20
pe = cor_rand(d) |> m -> CorMat{Pearson}(m)
margins = [Exponential(rand()*3) for _ in 1:d]

cor_adjust(pe, margins)

# cor_adjust(pe, margins)
