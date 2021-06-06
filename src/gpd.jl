module gpd
export gpdfit, gpd_quantile


"""
    gpdfit(x::AbstractArray, wip::Bool = true, min_grid_pts::Int = 30, wip = true, sort_x::Bool = true)

Given a sample \$x\$, estimate the parameters \$k\$ and \$\\sigma\$ of the generalized Pareto 
distribution (GPD), assuming the location parameter is 0. The fit uses
a weak prior for \$k\$, which will stabilize estimates for very small sample 
sizes (and low effective sample sizes in the case of MCMC samples). 
The weakly informative prior is a Gaussian centered at 0.5. 

# Arguments
- `x::AbstractArray`: A numeric vector. The sample from which to estimate the parameters.
- `wip::Bool = true`: Logical indicating whether to adjust k based on a weakly
  informative Gaussian prior centered on 0.5. Defaults to true.
- `min_grid_pts::Int = 30`: The minimum number of grid points used in the fitting
  algorithm. The actual number used is `min_grid_pts + floor(sqrt(length(x)))`.
- `sort_x::Bool = true`: If `true`, the first step in the fitting
  algorithm is to sort the elements of `x`. If `x` is already
  sorted in ascending order then `sort_x` can be set to `FALSE` to
  skip the initial sorting step.
- A named list with components `k` and `sigma`.

# Note
The parameter \$k\$ is the negative of \$k\$ in [zhangNewEfficientEstimation2009](@cite).
A slightly different quantile interpolation is used than in the paper.
"""
function gpdfit(x::AbstractArray, wip::Bool = true, min_grid_pts::Int = 30, sort_x::Bool = true)
    
    # x must be sorted, but we can skip if x is already sorted
    if sort_x
        sort!(x)
    end

    n = length(x)
    m = min_grid_pts + isqrt(n) # isqrt = floor sqrt
    prior = 3
    n_0 = 10  # determines how strongly to nudge kHat towards .5
    largest = maximum(x)
    quartile = quantile(x, .25; sorted = true, alpha = 0) 
    
    
    # build pointwise estimates of k and θ by using each element of the sample.
    vectorθ = @. 1 / x[n] + (1 - sqrt(m / (collect(1:m) - .5))) / prior / quartile
    vectorK = mean(log1p.(- vectorθ .* x'), dims = 2)  # take mean of each row
    logLikelihood = @. log(- vectorθ / vectorK) - vectorK - 1  # Calculate log-likelihood at each estimate
    weights = @. 1 / sum(exp(logLikelihood - logLikelihood')) # Calculate weights from log-likelihood

    θHat = weights ⋅ vectorθ  # Take the dot product of weights and pointwise estimates of θ to get the full estimate

    kHat = mean(log1p.(- θHat .* x))
    σ = -kHat / θHat
    # Drag towards .5 to reduce variance for small n
    if wip
      kHat = (kHat * n + .5 * n_0) / (n + n_0) 
    end

    return kHat, σ

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
  return sigma * expm1(-k * log1p(-p)) / k
end
