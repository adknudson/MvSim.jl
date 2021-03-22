module Bigsimr


using Base.Threads: @threads
using Distributions
using FastGaussQuadrature: gausshermite
using HypergeometricFunctions: _₂F₁
using IntervalArithmetic: interval, mid
using IntervalRootFinding: roots, Krawczyk
using IterTools: subsets
using LinearAlgebra
using LsqFit: curve_fit, coef
using PDMats
using Polynomials: Polynomial, derivative
using Printf
using QuadGK: quadgk
using SharedArrays
using SpecialFunctions: erfc, erfcinv
using StatsBase: corspearman, corkendall


import Statistics: cor


struct ValidCorrelationError <: Exception end

abstract type AbstractCorrelation end
struct Pearson  <: AbstractCorrelation end
struct Spearman <: AbstractCorrelation end
struct Kendall  <: AbstractCorrelation end
struct Adjusted <: AbstractCorrelation end


export AbstractCorrelation, Pearson, Spearman, Kendall, Adjusted
export CorMat, cortype
export MvDist
export cor, cor_threaded
export iscorrelation
export cor_rand
export cor_bounds
export cor_adjust
export cor_convert, cor_convert!
export cor_near_posdef, cor_fast_posdef, cor_fast_posdef!
export rand, rmvn, rvec


const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution

const CorOrNothing = Union{AbstractCorrelation, Nothing}
const PeSpKe = Union{Pearson, Spearman, Kendall}
const SpKe = Union{Spearman, Kendall}

const sqrt2 = sqrt(2)
const invsqrt2 = inv(sqrt(2))
const invsqrtpi = inv(sqrt(π))
const invsqrt2π = inv(sqrt(2π))


include("common.jl")

include("CorMat.jl")
include("GSDist.jl")
include("MvDist.jl")

include("rmvn.jl")
include("rvec.jl")

include("cor_adjust.jl")
include("cor_bounds.jl")
include("cor_convert.jl")
include("cor_fast_posdef.jl")
include("cor_near_posdef.jl")
include("cor.jl")
include("cor_rand.jl")

end
