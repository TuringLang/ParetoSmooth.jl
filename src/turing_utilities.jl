using .Turing, .MCMCChains
export pointwise_log_likelihoods

"""
    pointwise_log_likelihoods(chain::Chains, model)

Computes the pointwise log likelihoods from Turing model where [d,s,c] corresponds to 
log likelihood of evaluated at datapoint d, sample s, for chain c. Note that
currently the posterior log likelihood must be computed in a for loop in the 
Turing model.

# Arguments
- `chain::Chains`: a chain object from MCMCChains from a Turing model
- `model`: a Turing model with data in the form of model(data)

# Returns
- `Array{Float64,3}`: a three dimensional array of pointwise log likelihoods

"""
function pointwise_log_likelihoods(chain::Chains, model)
    # subset of chain for mcmc samples
    chain_params = MCMCChains.get_sections(chain, :parameters)
    # compute the pointwise log likelihoods
    pointwise_log_like_dict = pointwise_loglikelihoods(model, chain_params)
    # parse "var[i]" -> i
    ind_from_string(x) = parse(Int, split(split(x, "[")[2], "]")[1])
    # collect variable names
    sorted_keys = sort(collect(keys(pointwise_log_like_dict)); by=ind_from_string)
    # Convert from dictionary to 3d array
    return permutedims(cat((pointwise_log_like_dict[k] for k in sorted_keys)...; dims=3), (3, 1, 2))
end

"""
    function psis_loo(ll_fun::Function, chain::Chains, data, args...;
        source::String="mcmc" [, chain_index::Vector{Int}, kwargs...]
    ) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score based on MCMCChain object and Turing model. Note that the Turing model must compute 
the log likelihood in a loop over the data rather than using broadcasting. Otherwise the 
correct pointwise log likehoods will not be used.

# Arguments

  - `chain::Chains`: a chain object from MCMCChains from a Turing model
  - `model`: a Turing model with data in the form of model(data)
  - `args...`: Positional arguments to be passed to [`psis`](@ref).
  - `chain_index::Vector`: A vector of integers specifying which chain each iteration belongs to. For
    instance, `chain_index[iteration]` should return `2` if `log_likelihood[:, step]`
    belongs to the second chain.
  - `kwargs...`: Keyword arguments to be passed to [`psis`](@ref).

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function psis_loo(
    model, chain::Chains, args...;
    source::Union{AbstractString,Symbol}="mcmc", log_weights::Bool=false, kwargs...
    )
    pointwise_log_likes = pointwise_log_likelihoods(chain, model)
    return psis_loo(pointwise_log_likes, args...; source, log_weights, kwargs...)
end
"""
    psis(
        ll_fun::Function, 
        chain::Chains,
        data; 
        source::String="mcmc", 
        log_weights::Bool=false
    ) -> Psis

Implements Pareto-smoothed importance sampling (PSIS) based on MCMCChain object and Turing model.
Note that the Turing model must compute the log likelihood in a loop over the data rather than 
using broadcasting. Otherwise the correct pointwise log likehoods will not be used.

# Arguments
## Positional Arguments
- `chain::Chains`: a chain object from MCMCChains from a Turing model
- `model`: a Turing model with data in the form of model(data)
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
    model, chain::Chains, r_eff;
    source::Union{AbstractString,Symbol}="mcmc", log_weights::Bool=false
    )
    pointwise_log_likes = pointwise_log_likelihoods(chain, model)
    return psis(-pointwise_log_likes, r_eff; source, log_weights)
end

function psis(
    model, chain::Chains;
    source::Union{AbstractString,Symbol}="mcmc", log_weights::Bool=false
    )
    pointwise_log_likes = pointwise_log_likelihoods(chain, model)
    r_eff = similar(pointwise_log_likes, 0)
    return psis(-pointwise_log_likes, r_eff; source, log_weights)
end