"""
    cor(x[, y], ::Union{Pearson, Spearman, Kendall})

Compute the correlation matrix of a given type.

The possible correlation types are:
* `Pearson`
* `Spearman`
* `Kendall`

# Examples
```jldoctest
julia> x = [-1.62169     0.0158613   0.500375  -0.794381
             2.50689     3.31666    -1.3049     2.16058
             0.495674    0.348621   -0.614451  -0.193579
             2.32149     2.18847    -1.83165    2.08399
            -0.0573697   0.39908     0.270117   0.658458
             0.365239   -0.321493   -1.60223   -0.199998
            -0.55521    -0.898513    0.690267   0.857519
            -0.356979   -1.03724     0.714859  -0.719657
            -3.38438    -1.93058     1.77413   -1.23657
             1.57527     0.836351   -1.13275   -0.277048];

julia> cor(x, Pearson())
4×4 Array{Float64,2}:
  1.0        0.86985   -0.891312   0.767433
  0.86985    1.0       -0.767115   0.817407
 -0.891312  -0.767115   1.0       -0.596762
  0.767433   0.817407  -0.596762   1.0

julia> cor(x, Spearman())
4×4 Array{Float64,2}:
  1.0        0.866667  -0.854545   0.709091
  0.866667   1.0       -0.781818   0.684848
 -0.854545  -0.781818   1.0       -0.612121
  0.709091   0.684848  -0.612121   1.0

julia> cor(x, Kendall())
4×4 Array{Float64,2}:
  1.0        0.733333  -0.688889   0.555556
  0.733333   1.0       -0.688889   0.555556
 -0.688889  -0.688889   1.0       -0.422222
  0.555556   0.555556  -0.422222   1.0
```
"""
function cor end
cor(x,    ::Pearson)  = cor(x)
cor(x, y, ::Pearson)  = cor(x, y)
cor(x,    ::Spearman) = corspearman(x)
cor(x, y, ::Spearman) = corspearman(x, y)
cor(x,    ::Kendall)  = corkendall(x)
cor(x, y, ::Kendall)  = corkendall(x, y)


"""
    cor_threaded(x[, y], ::Union{Pearson, Spearman, Kendall})
"""
cor_threaded(x,    C::PeSpKe) = cor(x,    C)
cor_threaded(x, y, C::PeSpKe) = cor(x, y, C)
