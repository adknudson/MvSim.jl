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
using LinearAlgebra
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
struct Adjusted <: AbstractCorrelation end


export CorMat, MvDist



const UD  = UnivariateDistribution
const CUD = ContinuousUnivariateDistribution
const DUD = DiscreteUnivariateDistribution

const CorOrNothing = Union{AbstractCorrelation, Nothing}
const SpKe = Union{Spearman, Kendall}

const sqrt2 = sqrt(2)
const invsqrt2 = inv(sqrt(2))
const invsqrtpi = inv(sqrt(π))
const invsqrt2π = inv(sqrt(2π))


include("CorMat.jl")
include("MvDist.jl")
include("GSDist.jl")

include("precompile.jl")


end
