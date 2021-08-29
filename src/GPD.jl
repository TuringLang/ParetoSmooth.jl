using LinearAlgebra
using LoopVectorization
using Statistics
using TensorOperations
using Tullio


"""
    gpdfit(
        sample::AbstractVector{T<:Real}; 
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

Estimation method taken from Zhang, J. and Stephens, M.A. (2009). The parameter ξ is the
negative of k.
"""
function gpdfit(
    sample::AbstractVector{T};
    wip::Bool=true,
    min_grid_pts::Integer=30,
    sort_sample::Bool=false,
) where {T <: Real}

    len = length(sample)
    # sample must be sorted, but we can skip if sample is already sorted
    if sort_sample
        sample = sort(sample; alg=QuickSort)
    end


    grid_size = min_grid_pts + isqrt(len)  # isqrt = floor sqrt
    n_0 = 10  # determines how strongly to nudge ξ towards .5
    x_star::T = inv(3 * sample[(len + 2) ÷ 4])  # magic number. ¯\_(ツ)_/¯


    # build pointwise estimates of ξ and θ at each grid point
    θ_hats = similar(sample, grid_size)
    ξ_hats = similar(sample, grid_size)
    invmax = inv(sample[len])
    @tullio threads=false θ_hats[i] = invmax + (1 - sqrt((grid_size + 1) / i)) * x_star
    @tullio threads=false ξ_hats[i] = log1p(-θ_hats[i] * sample[j]) |> _ / len
    log_like = similar(ξ_hats)
    # Calculate profile log-likelihood at each estimate:
    @tullio threads=false log_like[i] =
        len * (log(-θ_hats[i] / ξ_hats[i]) - ξ_hats[i] - 1)
    # Calculate weights from log-likelihood:
    weights = ξ_hats  # Reuse preallocated array
    @tullio threads=false weights[y] = exp(log_like[x] - log_like[y]) |> inv
    # Take weighted mean:
    @tullio threads=false θ_hat := weights[x] * θ_hats[x]
    @tullio threads=false ξ := log1p(-θ_hat * sample[i])
    ξ /= len
    σ::T = -ξ / θ_hat

    # Drag towards .5 to reduce variance for small len
    if wip
        @fastmath ξ = (ξ * len + 0.5 * n_0) / (len + n_0)
    end

    return ξ::T, σ::T

end


"""
    gpd_quantile(p::T, k::T, sigma::T) where {T<:Real} -> T

Compute the `p` quantile of the Generalized Pareto Distribution (GPD).

# Arguments

  - `p`: A scalar between 0 and 1.
  - `ξ`: A scalar shape parameter.
  - `σ`: A scalar scale parameter.

# Returns

A quantile of the Generalized Pareto Distribution.
"""
function gpd_quantile(p, ξ::T, sigma::T) where {T <: Real}
    return sigma * expm1(-ξ * log1p(-p)) / ξ
end
