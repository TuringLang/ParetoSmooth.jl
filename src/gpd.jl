module gpd
export gpdfit, gpd_quantile

using Memoization, Statistics, LinearAlgebra, Tullio, LoopVectorization, KernelAbstractions


"""
    gpdfit(sample::AbstractArray, wip::Bool = true, min_grid_pts::Int = 30, wip = true, sort_sample::Bool = true)

Given a sample, estimate the parameters \$ξ\$ and \$\\sigma\$ of the  
generalized Pareto distribution (GPD), assuming the location parameter
is 0. The fit uses a weak prior for \$ξ\$, which will stabilize estimates 
for very small sample sizes (and low effective sample sizes in the case of 
MCMC samples). The weakly informative prior is a Gaussian centered at 0.5. 


# Arguments
- `sample::AbstractArray`: A numeric vector. The sample from which to estimate the parameters.
- `wip::Bool = true`: Logical indicating whether to adjust \$ξ\$ based on a weakly
  informative Gaussian prior centered on 0.5. Defaults to true.
- `min_grid_pts::Int = 30`: The minimum number of grid points used in the fitting
  algorithm. The actual number used is `min_grid_pts + floor(sqrt(length(sample)))`.
- `sort_sample::Bool = true`: If `true`, the first step in the fitting
  algorithm is to sort the elements of `sample`. If `sample` is already
  sorted in ascending order then `sort_sample` can be set to `false` to
  skip the initial sorting step.


# Returns
- `ξ, σ`: The estimated parameters of the generalized Pareto distribution.

# Note
The parameter \$ξ\$ is the negative of \$k\$ in [zhangNewEfficientEstimation2009](@cite).
A slightly different quantile interpolation is used than in the paper. This function will
modify the provided sample by sorting it to speed up future applications of gpdfit.
"""
function gpdfit(sample::AbstractVector, wip::Bool = true, min_grid_pts::Int = 30, sort_sample::Bool = true)
  
  n = length(sample)

  # sample must be sorted, but we can skip if sample is already sorted
  if sort_sample
      sort!(sample)
  end

  
  prior = 3.0
  m = min_grid_pts + isqrt(n) # isqrt = floor sqrt
  n_0 = 10.0  # determines how strongly to nudge ξ towards .5
  quartile = sample[(n+2) ÷ 4] 
  seq = collect(1:m) 


  # build pointwise estimates of k and θ by using each element of the sample.
  @turbo θHats = @. 1 / sample[n] + (1 - sqrt(m / (seq - .5))) / prior / quartile
  @tullio grad=false ξHats[x] := log1p(- θHats[x] * sample[y]) |> (_ / n)
  @turbo logLikelihood = @. n * (log(- θHats / ξHats) - ξHats - 1)  # Calculate log-likelihood at each estimate
  @tullio grad=false weights[y] := exp(logLikelihood[x] - logLikelihood[y]) |> inv # Calculate weights from log-likelihood

  θHat = weights ⋅ θHats  # Take the dot product of weights and pointwise estimates of θ to get the full estimate

  ξ = mean(log1p.(- θHat .* sample))
  σ = -ξ / θHat
  # Drag towards .5 to reduce variance for small n
  if wip
    ξ = (ξ * n + .5 * n_0) / (n + n_0) 
  end

  return ξ, σ

end

"""
  gpd_quantile(p::Real, k::Real, sigma::Real)

# Arguments  
- p A float between 0 and 1.
- k Scalar shape parameter.
- sigma Scalar scale parameter.
@return Vector of quantiles.
"""
function gpd_quantile(p::Real, k::Real, sigma::Real)
  return @fastmath sigma * expm1(-k * log1p(-p)) / k
end


end
