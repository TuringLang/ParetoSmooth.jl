using LinearAlgebra
using LogExpFunctions
using Statistics

"""
    gpd_fit(
        sample::AbstractVector{T<:Real},
        r_eff::T = 1; 
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
function gpd_fit(
    sample::AbstractVector{T},
    r_eff::T=1;
    wip::Bool=true,
    min_grid_pts::Integer=30,
    sort_sample::Bool=false,
) where T<:Real

    len = length(sample)
    # sample must be sorted, but we can skip if sample is already sorted
    if sort_sample
        sample = sort(sample; alg=QuickSort)
    end


    grid_size = min_grid_pts + isqrt(len)  # isqrt = floor sqrt
    n_0 = 10  # determines how strongly to nudge ξ towards .5
    x_star = inv(3 * sample[(len + 2) ÷ 4])  # magic number. ¯\_(ツ)_/¯
    invmax = inv(sample[len])

    # build pointwise estimates of ξ and θ at each grid point
    θ_hats = similar(sample, grid_size)
    @fastmath @. θ_hats = invmax + (1 - sqrt((grid_size + 1) / $(1:grid_size))) * x_star
    ξ_hats = similar(θ_hats)
    for i = eachindex(ξ_hats, θ_hats)
        ξ_hat = zero(eltype(ξ_hats))
        for j = eachindex(sample)
            ξ_hat += log1p(-θ_hats[i] * sample[j])
        end
        ξ_hats[i] = ξ_hat/len
    end

    log_like = ξ_hats  # Reuse preallocated array
    # Calculate profile log-likelihood at each estimate:
    for i = eachindex(ξ_hats, θ_hats)
        ξ_hats[i] = len * (log(-θ_hats[i] / ξ_hats[i]) - ξ_hats[i] - 1)
    end
    # Calculate weights from log-likelihood:
    weights = log_like  # Reuse preallocated array
    log_norm = logsumexp(log_like)
    for i = eachindex(log_like)
        log_like[i] = exp_inline(log_like[i] - log_norm)
    end
    # Take weighted mean:
    θ_hat = zero(Base.promote_eltype(θ_hats, weights))
    @simd for i = eachindex(θ_hats, weights)
        θ_hat += θ_hats[i] * weights[i]
    end
    ξ = zero(θ_hat)
    @simd for i = eachindex(sample)
        ξ += log1p(-θ_hat * sample[i])
    end
    ξ /= len
    σ::T = -ξ / θ_hat

    # Drag towards .5 to reduce variance for small len
    if wip
        @fastmath ξ = (r_eff * ξ * len + 0.5 * n_0) / (r_eff * len + n_0)
    end

    return ξ, σ

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
