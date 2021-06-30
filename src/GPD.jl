module GPD

using LinearAlgebra
using LoopVectorization
using Statistics
using Tullio



"""
    gpdfit(
        sample::AbstractVector{T<:AbstractFloat}; 
        wip::Bool=true, 
        min_grid_pts::Integer=30, 
        sort_sample::Bool=false
        ) -> (ξ::T, σ::T)

Return a named list of estimates for the parameters ξ (shape) and σ (scale) of the 
generalized Pareto distribution (GPD), assuming the location parameter is 0.

# Arguments

  - `sample::AbstractVector`: A numeric vector. The sample from which to estimate 
  the parameters.
  - `wip::Bool = true`: Logical indicating whether to adjust ξ based on a weakly informative 
  Gaussian prior centered on 0.5. Defaults to `true`.
  - `min_grid_pts::Integer = 30`: The minimum number of grid points used in the fitting 
  algorithm. The actual number used is `min_grid_pts + ⌊sqrt(length(sample))⌋`.

# Note

Estimation method taken from Zhang, J. and Stephens, M.A. (2009). The parameter 
ξ is the negative of \$k\$.
"""
function gpdfit(
    sample::AbstractVector{T};
    wip::Bool = true,
    min_grid_pts::Integer = 30,
    sort_sample::Bool = false,
) where T<:AbstractFloat

    len = length(sample)
    # sample must be sorted, but we can skip if sample is already sorted
    if sort_sample
        sample = sort(sample; alg = QuickSort)
    end


    prior = 3
    m = min_grid_pts + isqrt(len) # isqrt = floor sqrt
    n_0 = 10  # determines how strongly to nudge ξ towards .5
    quartile::T = sample[(len+2)÷4]


    # build pointwise estimates of ξ and θ at each grid point
    θHats = similar(sample, m)
    ξHats = similar(sample, m)
    @turbo @. θHats = 1 / sample[len] + (1 - sqrt((m+1) / $(1:m))) / prior / quartile
    @tullio threads=false ξHats[x] := log1p(-θHats[x] * sample[y]) |> _ / len
    logLikelihood = similar(ξHats)
    # Calculate profile log-likelihood at each estimate:
    @turbo @. logLikelihood = len * (log(-θHats / ξHats) - ξHats - 1) 
    # Calculate weights from log-likelihood:
    weights = ξHats  # Reuse preallocated array (which is no longer in use)
    @tullio threads=false weights[y] = exp(logLikelihood[x] - logLikelihood[y]) |> inv
    θHat::T = zero(T)
    # Take weighted mean:
    @tullio threads=false θHat = weights[x] * θHats[x]

    ξ::T = calc_ξ(sample, θHat)

    σ::T = -ξ / θHat
    # Drag towards .5 to reduce variance for small len
    if wip
        ξ = (ξ * len + 0.5 * n_0) / (len + n_0)
    end

    return ξ::T, σ::T

end

"""
    gpd_quantile(p::T, k::T, sigma::T) where {T<:AbstractFloat} -> T

Compute the `p` quantile of the Generalized Pareto Distribution (GPD).

# Arguments

  - `p`: A scalar between 0 and 1.
  - `ξ`: A scalar shape parameter.
  - `σ`: A scalar scale parameter.

# Returns

A quantile of the Generalized Pareto Distribution.
"""
function gpd_quantile(p, k::T, sigma::T) where {T<:AbstractFloat}
    quant = sigma * expm1(-k * log1p(-p)) / k
    return quant
end


"""
    calc_ξ(sample, θHat)

Calculate ξ, the parameter for the GPD.
"""
function calc_ξ(sample::AbstractVector{T}, θHat::T) where {T<:AbstractFloat}
    ξ = zero(T)
    @turbo for i in eachindex(sample)
        ξ += log1p(-θHat * sample[i]) / length(sample)
    end
    return ξ::T
end


end
