module Bigsimr


using Base.Threads: @threads
using Distributions
using Distributions: UnivariateDistribution, ContinuousUnivariateDistribution,
    DiscreteUnivariateDistribution
using FastGaussQuadrature: gausshermite
using HypergeometricFunctions: _₂F₁
using IntervalArithmetic: interval, mid
using IntervalRootFinding: roots, Krawczyk
using IterTools: subsets
using LinearAlgebra: Cholesky, Diagonal, Symmetric
using LinearAlgebra: cholesky, diag, diagind, diagm, eigen, isposdef, issymmetric, norm
using LsqFit: curve_fit, coef
using PDMats
using Polynomials: Polynomial, derivative
using QuadGK: quadgk
using SharedArrays
using SpecialFunctions: erfc, erfcinv
using Statistics: clampcor
using StatsBase: corspearman, corkendall


import Statistics: cor


struct ValidCorrelationError <: Exception end

abstract type AbstractCorrelation end
struct Pearson  <: AbstractCorrelation end
struct Spearman <: AbstractCorrelation end
struct Kendall  <: AbstractCorrelation end


export rvec, rmvn
export pearson_match, pearson_bounds
export AbstractCorrelation, Pearson, Spearman, Kendall
export cor, cor_fast
export cor_nearPD, cor_fastPD, cor_fastPD!
export cor_randPD, cor_randPSD
export cor_convert, cor_bounds, cor_constrain, cor_constrain!
export cov2cor, cov2cor!, clamp, cor_clamp
export iscorrelation


const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution

const sqrt2 = sqrt(2)
const invsqrt2 = inv(sqrt(2))
const invsqrtpi = inv(sqrt(π))
const invsqrt2π = inv(sqrt(2π))


include("utils.jl")

include("RandomVector/rvec.jl")
include("RandomVector/rmvn.jl")
include("RandomVector/utils.jl")

include("Correlation/cor_bounds.jl")
include("Correlation/fast_pos_def.jl")
include("Correlation/nearest_pos_def.jl")
include("Correlation/random.jl")
include("Correlation/utils.jl")

include("PearsonMatching/pearson_match.jl")
include("PearsonMatching/pearson_bounds.jl")
include("PearsonMatching/utils.jl")

include("GSDist/GSDist.jl")
include("GSDist/utils.jl")

include("precompile.jl")


end
