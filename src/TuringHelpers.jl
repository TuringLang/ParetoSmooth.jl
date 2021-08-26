using .Turing
export pointwise_log_likelihoods, psis_loo, psis


const TURING_MODEL_ARG = """
`model`: A Turing model with data in the form of `model(data)`.
"""


"""
    pointwise_log_likelihoods(model::DynamicPPL.Model, chain::Chains)

Compute the pointwise log-likelihoods from a Turing model.  

# Arguments
  - $TURING_MODEL_ARG
  - $CHAINS_ARG

# Returns
  - `Array`: A three dimensional array of pointwise log likelihoods. This array should be
    indexed using `array[data, sample, chain]`.
"""
function pointwise_log_likelihoods(model::DynamicPPL.Model, chain::Chains)
    # subset of chain for mcmc samples
    chain_params = MCMCChains.get_sections(chain, :parameters)
    # compute the pointwise log likelihoods
    log_like_dict = DynamicPPL.pointwise_loglikelihoods(model, chain_params)
    # Size of array (n_steps, n_chains) using first parameter
    dims = size(last(first(log_like_dict)))
    # parse "var[i]" -> i
    ind_from_string(x) = parse(Int, split(split(x, "[")[2], "]")[1])
    # collect variable names
    sorted_keys = sort(collect(keys(log_like_dict)); by=ind_from_string)
    # Convert from dictionary to 3d array
    array = [reshape(log_like_dict[i], 1, dims...) for i in sorted_keys]
    return reduce(vcat, array)
end


"""
    function psis_loo(
        ll_fun::Function, 
        chains::Chains
        [, args...; kwargs...]
    ) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score from a `chains` object and a Turing model. 

# Arguments

  - $CHAINS_ARG
  - $TURING_MODEL_ARG
  - $ARGS [`psis`](@ref).
  - $KWARGS [`psis`](@ref).

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function psis_loo(model::DynamicPPL.Model, chain::Chains, args...; kwargs...)
    pointwise_log_likes = pointwise_log_likelihoods(model, chain)
    return psis_loo(pointwise_log_likes, args...; kwargs...)
end


"""
    psis(
        model::DynamicPPL.Model, 
        chains::Chains,
        args...; 
        kwargs...
    ) -> Psis

Generate samples using Pareto smoothed importance sampling (PSIS).

# Arguments
  - $TURING_MODEL_ARG
  - $CHAINS_ARG
  - $ARGS [`psis`](@ref).
  - $KWARGS [`psis`](@ref).

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function psis(model::DynamicPPL.Model, chain::Chains, args...; kwargs...)
    log_ratios = pointwise_log_likelihoods(model, chain)
    return psis(-log_ratios, args...; kwargs...)
end


"""
    naive_lpd(model::DynamicPPL.Model, chain::Chains, args...; kwargs...)

Calculate the naive (in-sample) estimate of the log probability density, otherwise
known as the Bayes score. Not recommended for most uses.
"""
function naive_lpd(model::DynamicPPL.Model, chain::Chains, args...; kwargs...)
    log_ratios = pointwise_log_likelihoods(model, chain)
    return naive_lpd(log_ratios, args...; kwargs...)
end
