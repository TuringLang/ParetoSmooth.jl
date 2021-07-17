using .MCMCChains
export pointwise_log_likelihoods

"""
    pointwise_log_likelihoods(ll_fun::Function, chain::Chains, data)

Computes the pointwise log likelihoods where [d,s,c] corresponds to log likelihood of 
evaluated at datapoint d, sample s, for chain c. 

# Arguments
- `ll_fun::Function`: a function that computes the log likelihood of a single data point: 
    f(θ1, ..., θn, data), where θi is the ith parameter
- `chain::Chain`: a chain object from MCMCChains
- `data`: a vector of data used to estimate parameters of the model

# Returns
- `Array{Float64,3}`: a three dimensional array of pointwise log likelihoods 
"""
function pointwise_log_likelihoods(ll_fun::Function, chain::Chains, data)
    samples = Array(Chains(chain, :parameters).value)
    pointwise_log_likelihoods(ll_fun, samples, data)
end

"""
    function psis_loo(ll_fun::Function, chain::Chains, data, args...;
        source::String="mcmc" [, chain_index::Vector{Int}, kwargs...]
    ) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score based on MCMCChain.

# Arguments

  - `ll_fun::Function`: a function that computes the log likelihood of a single data point: 
    f(θ1, ..., θn, data), where θi is the ith parameter
  - `chain::Chain`: a chain object from MCMCChains
  - `data`: a vector of data used to estimate parameters of the model
  - `args...`: Positional arguments to be passed to [`psis`](@ref).
  - `chain_index::Vector`: A vector of integers specifying which chain each iteration belongs to. For
    instance, `chain_index[iteration]` should return `2` if `log_likelihood[:, step]`
    belongs to the second chain.
  - `kwargs...`: Keyword arguments to be passed to [`psis`](@ref).

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function psis_loo(
    ll_fun::Function, chain::Chains, data, args...;
    source::Union{AbstractString,Symbol}="mcmc", log_weights::Bool=false, kwargs...
    )
    pointwise_log_likes = pointwise_log_likelihoods(ll_fun, chain, data)
    return psis_loo(-1 * pointwise_log_likes, args...; source, log_weights, kwargs...)
end

"""
    psis(
        ll_fun::Function, 
        chain::Chains,
        data; 
        source::String="mcmc", 
        log_weights::Bool=false
    ) -> Psis

Implements Pareto-smoothed importance sampling (PSIS) based on MCMCChain object.

# Arguments
## Positional Arguments
- `ll_fun::Function`: a function that computes the log likelihood of a single data point: 
f(θ1, ..., θn, data), where θi is the ith parameter
- `chain::Chain`: a chain object from MCMCChains
- `data`: a vector of data used to estimate parameters of the model
- `r_eff::AbstractArray{T}`: An (optional) vector of relative effective sample sizes used 
in ESS calculations. If left empty, calculated automatically using the FFTESS method 
from InferenceDiagnostics.jl. See `relative_eff` to calculate these values. 

## Keyword Arguments

- `chain_index::Vector{Integer}`: An (optional) vector of integers indicating which chain 
each sample belongs to.
- `source::String="mcmc"`: A string or symbol describing the source of the sample being 
used. If `"mcmc"`, adjusts ESS for autocorrelation. Otherwise, samples are assumed to be 
independent. Currently permitted values are $SAMPLE_SOURCES.
- `log_weights::Bool=false`: Return log weights, rather than the PSIS weights. 
"""
function psis(
    ll_fun::Function, chain::Chains, data, r_eff;
    source::Union{AbstractString,Symbol}="mcmc", log_weights::Bool=false
    )
    pointwise_log_likes = pointwise_log_likelihoods(ll_fun, chain, data)
    return psis(-1 * pointwise_log_likes, r_eff; source, log_weights)
end

function psis(
    ll_fun::Function, chain::Chains, data;
    source::Union{AbstractString,Symbol}="mcmc", log_weights::Bool=false
    )
    pointwise_log_likes = pointwise_log_likelihoods(ll_fun, chain, data)
    r_eff = similar(pointwise_log_likes, 0)
    return psis(-1 * pointwise_log_likes, r_eff; source, log_weights)
end
