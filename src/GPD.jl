module GPD

using LinearAlgebra
using LoopVectorization
using Statistics
using Tullio



"""
    gpdfit(
        sample::AbstractVector{F<:AbstractFloat}, 
        wip::Bool=true, 
        min_grid_pts::Integer=30, 
        sort_sample::Bool=false
        )::NamedTuple{}

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
    sample::AbstractVector;
    wip::Bool = true,
    min_grid_pts::Integer = 30,
    sort_sample::Bool = false,
)

    n = length(sample)
    # sample must be sorted, but we can skip if sample is already sorted
    if sort_sample
        sample = sort(sample; alg = QuickSort)
    end


    prior = 3
    m = min_grid_pts + isqrt(n) # isqrt = floor sqrt
    n_0 = 10  # determines how strongly to nudge ξ towards .5
    quartile = sample[(n+2)÷4]


    # build pointwise estimates of ξ and θ by using each element of the sample.
    @turbo θHats = @. 1 / sample[n] + (1 - sqrt((m + 1) / $(1:m))) / prior / quartile
    @tullio threads=false ξHats[x] := (log1p(-θHats[x] * sample[y])) |> _ / n
    logLikelihood = similar(ξHats)
    @turbo @. logLikelihood = n * (log(-θHats / ξHats) - ξHats - 1)  # Calculate log-likelihood at each estimate
    # Calculate weights from log-likelihood:
    weights = ξHats  # Reuse preallocated array
    @tullio threads=false weights[y] = inv(exp(logLikelihood[x] - logLikelihood[y]))
    # Take weighted mean:
    @tullio threads=false θHat := weights[x] * θHats[x]

    ξ = calc_ξ(sample, θHat)

    σ = -ξ / θHat
    # Drag towards .5 to reduce variance for small n
    if wip
        ξ = (ξ * n + 0.5 * n_0) / (n + n_0)
    end

    return ξ, σ

end

"""
    gpd_quantile(p::Real, k::Real, sigma::Real)

Compute the `p` quantile of the Generalized Pareto Distribution (GPD).

# Arguments

  - `p`: A scalar between 0 and 1.
  - `ξ`: A scalar shape parameter.
  - `σ`: A scalar scale parameter.

# Returns

A quantile of the Generalized Pareto Distribution.
"""
function gpd_quantile(p::Real, k::Real, sigma::Real)
    return @fastmath sigma * expm1(-k * log1p(-p)) / k
end


"""
    calc_ξ(sample, θHat)

Calculate ξ, the parameter for the GPD.
"""
function calc_ξ(sample::AbstractVector, θHat::Real)
    ξ = zero(promote_type(typeof(θHat), eltype(sample)))
    @turbo for i in eachindex(sample)
        ξ += log1p(-θHat * sample[i])
    end
    return ξ / length(sample)
end


end
