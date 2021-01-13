# Utilities

## General Utilities

```@docs
MvSim.hermite(x::Float64, n::Int, probabilists::Bool=true)
MvSim.normal_to_margin(d::UnivariateDistribution, x::Float64)
```

## Random Multivariate Vector Utilities

```@docs
MvSim._randn(n::Int, d::Int)
MvSim._rmvn(n::Int, ρ::Matrix{Float64})
```

## Pearson Matching Utilities

```@docs
MvSim.get_coefs(margin::UnivariateDistribution, n::Int)
MvSim.Hp(x::Float64, n::Int)
MvSim.Gn0d(n::Int, A::UnitRange{Int}, B::UnitRange{Int}, α::Vector{Float64}, β::Vector{Float64}, σAσB_inv::Float64)
MvSim.Gn0m(n::Int, A::UnitRange{Int}, α::Vector{Float64}, dB::UnivariateDistribution , σAσB_inv::Float64)
MvSim.solve_poly_pm_one(coef::Vector{Float64})
```

## Nearest Positive Definite Correlation Matrix Utilities

```@docs
MvSim.npd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64}, n::Int)
MvSim.npd_pca(b::Vector{Float64}, X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64}, n::Int)
MvSim.npd_pre_cg(b::Vector{Float64}, c::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, ϵ::Float64, N::Int, n::Int)
MvSim.npd_precond_matrix(Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)
MvSim.npd_set_omega(λ::Vector{Float64}, n::Int)
MvSim.npd_jacobian(x::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)
```

## Fast Positive Definite Correlation 

```@docs
MvSim.fast_pca!(X::Matrix{T}, λ::Vector{T}, P::Matrix{T}, n::Int) where T<:AbstractFloat
```