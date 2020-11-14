var documenterSearchIndex = {"docs":
[{"location":"utilities/#Utilities","page":"Utilities","title":"Utilities","text":"","category":"section"},{"location":"utilities/#General-Utilities","page":"Utilities","title":"General Utilities","text":"","category":"section"},{"location":"utilities/","page":"Utilities","title":"Utilities","text":"MvSim.hermite(x::Real, n::Int; probabilists::Bool=true)\nMvSim.setdiag(A::Matrix{<:Real}, x::Real)\nMvSim.normal_to_margin(d::UnivariateDistribution, x::AbstractArray)","category":"page"},{"location":"utilities/#MvSim.hermite-Tuple{Real,Int64}","page":"Utilities","title":"MvSim.hermite","text":"hermite(x, n::Int, probabilists::Bool=true)\n\nCompute the Hermite polynomials of degree n. Compute the Probabilists' version by default.\n\nThe two definitions of the Hermite polynomials are each a rescaling of the other. Let Heₙ(x) denote the Probabilists' version, and Hₙ(x) the Physicists'. Then\n\nH_n(x) = 2^fracn2 He_nleft(sqrt2 xright)\n\nHe_n(x) = 2^-fracn2 H_nleft(fracxsqrt2right)\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.setdiag-Tuple{Array{var\"#s1\",2} where var\"#s1\"<:Real,Real}","page":"Utilities","title":"MvSim.setdiag","text":"setdiag(A::Matrix{<:Real}, x::Real)\n\nSet the diagonal elements of a Matrix to a value. Return the new matrix.\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.normal_to_margin-Tuple{Distribution{Univariate,S} where S<:ValueSupport,AbstractArray}","page":"Utilities","title":"MvSim.normal_to_margin","text":"normal_to_margin(d::UnivariateDistribution, x)\n\nConvert samples from a standard normal distribution to a given marginal distribution.\n\n\n\n\n\n","category":"method"},{"location":"utilities/#Pearson-Matching-Utilities","page":"Utilities","title":"Pearson Matching Utilities","text":"","category":"section"},{"location":"utilities/","page":"Utilities","title":"Utilities","text":"MvSim.get_coefs(margin::UnivariateDistribution, n::Int)\nMvSim.Hϕ(x::Real, n::Int)\nMvSim.Gn0d(n::Int, A, B, α, β, σAσB_inv)\nMvSim.Gn0m(n::Int, A, α, dB, σAσB_inv)\nMvSim.solve_poly_pm_one(coef)","category":"page"},{"location":"utilities/#MvSim.get_coefs-Tuple{Distribution{Univariate,S} where S<:ValueSupport,Int64}","page":"Utilities","title":"MvSim.get_coefs","text":"get_coefs(margin::UnivariateDistribution, n::Int)\n\nGet the n^th degree Hermite Polynomial expansion coefficients for F^-1Φ() where F^-1 is the inverse CDF of a probability distribution and Φ(⋅) is the CDF of a standard normal distribution.\n\nNotes\n\nThe paper describes using Guass-Hermite quadrature using the Probabilists' version of the Hermite polynomials, while the package FastGaussQuadrature.jl uses the Physicists' version. Because of this, we need to do a rescaling of the input and the output:\n\nfrac1ksum_s=1^mw_s H_k (t_s) F_i^-1leftPhi(t_s)right \nfrac1sqrtpi cdot ksum_s=1^mw_s H_k (t_ssqrt2) F_i^-1leftPhi(t_s)right\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.Hϕ-Tuple{Real,Int64}","page":"Utilities","title":"MvSim.Hϕ","text":"Hϕ(x::T, n::Int) where T<:Real\n\nWe need to account for when x is ±∞ otherwise Julia will return NaN for 0×∞\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.Gn0d-Tuple{Int64,Any,Any,Any,Any,Any}","page":"Utilities","title":"MvSim.Gn0d","text":"Gn0d(::Int, A, B, α, β, σAσB_inv)\n\nCalculate the n^th derivative of G at 0 where ρ_x = G(ρ_z)\n\nWe are essentially calculating a double integral over a rectangular region\n\nint_α_r-1^α_r int_β_s-1^β_s Φ(z_i z_j ρ_z) dz_i dz_j\n\n(α[r], β[s+1]) +----------+ (α[r+1], β[s+1])\n               |          |\n               |          |\n               |          |\n  (α[r], β[s]) +----------+ (α[r+1], β[s])\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.Gn0m-Tuple{Int64,Any,Any,Any,Any}","page":"Utilities","title":"MvSim.Gn0m","text":"Gn0m(::Int, A, α, dB, σAσB_inv)\n\nCalculate the n^th derivative of G at 0 where ρ_x = G(ρ_z)\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.solve_poly_pm_one-Tuple{Any}","page":"Utilities","title":"MvSim.solve_poly_pm_one","text":"solve_poly_pm_one(coef)\n\nSolve a polynomial equation on the interval [-1, 1].\n\n\n\n\n\n","category":"method"},{"location":"utilities/#Nearest-Positive-Definite-Correlation-Matrix-Utilities","page":"Utilities","title":"Nearest Positive Definite Correlation Matrix Utilities","text":"","category":"section"},{"location":"utilities/","page":"Utilities","title":"Utilities","text":"MvSim.npd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64}, n::Int)\nMvSim.npd_pca(X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64}, n::Int)\nMvSim.npd_pre_cg(b::Vector{Float64}, c::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, ϵ::Float64, N::Int, n::Int)\nMvSim.npd_precond_matrix(Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)\nMvSim.npd_set_omega(λ::Vector{Float64}, n::Int)\nMvSim.npd_jacobian(x::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int; PERTURBATION::Float64=1e-9)","category":"page"},{"location":"utilities/#MvSim.npd_gradient-Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,2},Array{Float64,1},Int64}","page":"Utilities","title":"MvSim.npd_gradient","text":"npd_gradient(y::Vector{Float64}, λ₀::Vector{Float64}, P::Matrix{Float64}, b₀::Vector{Float64}, n::Int)\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.npd_pca-Tuple{Array{Float64,2},Array{Float64,1},Array{Float64,2},Int64}","page":"Utilities","title":"MvSim.npd_pca","text":"npd_pca(X::Matrix{Float64}, λ::Vector{Float64}, P::Matrix{Float64}, n::Int)\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.npd_pre_cg-Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,2},Array{Float64,2},Float64,Int64,Int64}","page":"Utilities","title":"MvSim.npd_pre_cg","text":"npd_pre_cg(b::Vector{Float64}, c::Vector{Float64}, Ω₀::Matrix{Float64}, P::Matrix{Float64}, ϵ::Float64, N::Int, n::Int)\n\nPre- Conjugate Gradient method.\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.npd_precond_matrix-Tuple{Array{Float64,2},Array{Float64,2},Int64}","page":"Utilities","title":"MvSim.npd_precond_matrix","text":"npd_precond_matrix(Ω₀::Matrix{Float64}, P::Matrix{Float64}, n::Int)\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.npd_set_omega-Tuple{Array{Float64,1},Int64}","page":"Utilities","title":"MvSim.npd_set_omega","text":"npd_set_omega(λ::Vector{Float64}, n::Int)\n\n\n\n\n\n","category":"method"},{"location":"utilities/#MvSim.npd_jacobian-Tuple{Array{Float64,1},Array{Float64,2},Array{Float64,2},Int64}","page":"Utilities","title":"MvSim.npd_jacobian","text":"npd_jacobian(x, Ω₀, P, n; PERTURBATION=1e-9)\n\n\n\n\n\n","category":"method"},{"location":"#MvSim.jl-Documentation","page":"MvSim.jl Documentation","title":"MvSim.jl Documentation","text":"","category":"section"},{"location":"","page":"MvSim.jl Documentation","title":"MvSim.jl Documentation","text":"Pages = [\"main_functions.md\", \"utilities.md\"]","category":"page"},{"location":"#Index","page":"MvSim.jl Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"MvSim.jl Documentation","title":"MvSim.jl Documentation","text":"","category":"page"},{"location":"main_functions/#Main-Functions","page":"Main Functions","title":"Main Functions","text":"","category":"section"},{"location":"main_functions/#Random-Multivariate-Vector","page":"Main Functions","title":"Random Multivariate Vector","text":"","category":"section"},{"location":"main_functions/","page":"Main Functions","title":"Main Functions","text":"MvDistribution\nrvec\nBase.rand(D::MvDistribution, n::Int)","category":"page"},{"location":"main_functions/#MvSim.MvDistribution","page":"Main Functions","title":"MvSim.MvDistribution","text":"MvDistribution(R, margins, C)\n\nSimple data structure for storing a multivariate mixed distribution.\n\n\n\n\n\n","category":"type"},{"location":"main_functions/#MvSim.rvec","page":"Main Functions","title":"MvSim.rvec","text":"rvec(n, margins, ρ)\n\nGenerate samples for a list of marginal distributions and a correaltion structure.\n\n\n\n\n\n","category":"function"},{"location":"main_functions/#Base.rand-Tuple{MvDistribution,Int64}","page":"Main Functions","title":"Base.rand","text":"Base.rand(D::MvDistribution, n::Int)\n\nMore general wrapper for rvec.\n\n\n\n\n\n","category":"method"},{"location":"main_functions/#Correlations","page":"Main Functions","title":"Correlations","text":"","category":"section"},{"location":"main_functions/","page":"Main Functions","title":"Main Functions","text":"cor\ncor_convert\ncor_nearPD(R::Matrix{Float64};\n    τ::Float64=1e-5,\n    iter_outer::Int=200,\n    iter_inner::Int=20,\n    N::Int=200,\n    err_tol::Float64=1e-6,\n    precg_err_tol::Float64=1e-2,\n    newton_err_tol::Float64=1e-4)\ncor_nearPSD(A::Matrix{T}; n_iter::Int=100) where {T<:Real}\ncor_randPD\ncor_randPSD","category":"page"},{"location":"main_functions/#Statistics.cor","page":"Main Functions","title":"Statistics.cor","text":"cor(x, ::Type{<:Correlation})\n\nCompute the correlation matrix. The possible correlation     types are Pearson, Spearman, or Kendall.\n\n\n\n\n\n","category":"function"},{"location":"main_functions/#MvSim.cor_convert","page":"Main Functions","title":"MvSim.cor_convert","text":"cor_convert(ρ::Real, from::Correlation, to::Correlation)\n\nConvert from one type of correlation matrix to another. The possible correlation types are Pearson, Spearman, or Kendall.\n\n\n\n\n\n","category":"function"},{"location":"main_functions/#MvSim.cor_nearPD-Tuple{Array{Float64,2}}","page":"Main Functions","title":"MvSim.cor_nearPD","text":"cor_nearPD(R::Matrix{Float64};\n    τ::Float64=1e-5,\n    iter_outer::Int=200,\n    iter_inner::Int=20,\n    N::Int=200,\n    err_tol::Float64=1e-6,\n    precg_err_tol::Float64=1e-2,\n    newton_err_tol::Float64=1e-4)\n\nCompute the nearest positive definite correlation matrix given a symmetric correlation matrix R. This algorithm is based off of work by Qi and Sun 2006. Matlab, C, R, and Python code can be found on Sun's page. The algorithm has also been implemented in Fortran in the NAG library.\n\nArguments\n\nτ::Float64: a [small] nonnegative number used to enforce a minimum eigenvalue.\nerr_tol::Float64: the error tolerance for the stopping condition.\n\nExamples\n\nimport LinearAlgebra: eigvals\n# Define a negative definite correlation matrix\nρ = [1.00 0.82 0.56 0.44\n     0.82 1.00 0.28 0.85\n     0.56 0.28 1.00 0.22\n     0.44 0.85 0.22 1.00]\neigvals(ρ)\n\nr = cor_nearPD(ρ)\neigvals(r)\n\n\n\n\n\n","category":"method"},{"location":"main_functions/#MvSim.cor_nearPSD-Union{Tuple{Array{T,2}}, Tuple{T}} where T<:Real","page":"Main Functions","title":"MvSim.cor_nearPSD","text":"cor_nearPSD(A::Matrix{T}; n_iter::Int=100) where {T<:Real}\n\n\n\n\n\n","category":"method"},{"location":"main_functions/#MvSim.cor_randPD","page":"Main Functions","title":"MvSim.cor_randPD","text":"cor_randPD(T::Type{<:AbstractFloat}, d::Int, α::Real=1.0)\n\nGenerate a random positive definite correlation matrix of size dd. The parameter α is used to determine the autocorrelation in the correlation coefficients.\n\nReference\n\nJoe H (2006). Generating random correlation matrices based on partial correlations. J. Mult. Anal. Vol. 97, 2177–2189.\n\n\n\n\n\n","category":"function"},{"location":"main_functions/#MvSim.cor_randPSD","page":"Main Functions","title":"MvSim.cor_randPSD","text":"cor_randPSD(T::Type{<:AbstractFloat}, d::Int, k::Int=d)\n\nCompute a random positive semidefinite correlation matrix\n\nReference\n\nhttps://stats.stackexchange.com/a/125020\n\n\n\n\n\n","category":"function"},{"location":"main_functions/#Pearson-Matching","page":"Main Functions","title":"Pearson Matching","text":"","category":"section"},{"location":"main_functions/","page":"Main Functions","title":"Main Functions","text":"ρz(ρx::Real, dA::UnivariateDistribution, dB::UnivariateDistribution; n::Int=7)\nρz_bounds","category":"page"},{"location":"main_functions/#MvSim.ρz-Tuple{Real,Distribution{Univariate,S} where S<:ValueSupport,Distribution{Univariate,S} where S<:ValueSupport}","page":"Main Functions","title":"MvSim.ρz","text":"ρz(ρx::Real, dA::UD, dB::UD; n::Int=7)\n\nCompute the pearson correlation coefficient that is necessary to achieve the target correlation given a pair of marginal distributions.\n\n\n\n\n\n","category":"method"},{"location":"main_functions/#MvSim.ρz_bounds","page":"Main Functions","title":"MvSim.ρz_bounds","text":"ρz_bounds\n\nCompute the lower and upper bounds of possible correlations for a pair of univariate distributions. The value n determines the accuracy of the  approximation of the two distributions.\n\n\n\n\n\n","category":"function"}]
}
