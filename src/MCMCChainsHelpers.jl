using .MCMCChains
export pointwise_log_likelihoods


const CHAINS_ARG = """
`chains::Chains`: A chain object from MCMCChains.
"""


"""
    pointwise_log_likelihoods(ll_fun::Function, chains::Chains, data)

Compute the pointwise log likelihoods.

# Arguments
  - $LIKELIHOOD_FUNCTION_ARG
  - $CHAINS_ARG
  - $DATA_ARG

# Returns
  - `Array`: a three dimensional array of pointwise log-likelihoods. Dimensions are ordered
    as `[data, step, chain]`.
"""
function pointwise_log_likelihoods(
    ll_fun::Function, chain::Chains, data::AbstractVector; kwargs...
)
    samples = Chains(chain, :parameters).value
    return pointwise_log_likelihoods(ll_fun, samples, data; kwargs...)
end


"""
    function psis_loo(
        ll_fun::Function, 
        chain::Chains, 
        data, 
        args...; 
        kwargs...
    ) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score from an MCMCChains object.

# Arguments

  - $LIKELIHOOD_FUNCTION_ARG
  - $CHAINS_ARG
  - $DATA_ARG
  - $ARGS [`psis_loo`](@ref).
  - $KWARGS [`psis_loo`](@ref).

See also: [`psis_loo`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function psis_loo(ll_fun::Function, chain::Chains, data::AbstractVector, args...; kwargs...)
    pointwise_log_likes = pointwise_log_likelihoods(ll_fun, chain, data)
    return psis_loo(pointwise_log_likes, args...; kwargs...)
end

"""
    psis(ll_fun::Function, chain::Chains, data[, kwargs...]; args...) -> Psis

Implements Pareto-smoothed importance sampling (PSIS) based on MCMCChain object.

# Arguments

  - $LIKELIHOOD_FUNCTION_ARG
  - $CHAINS_ARG
  - $DATA_ARG
  - $ARGS [`psis`](@ref).
  - $KWARGS [`psis`](@ref).

See also: [`psis`](@ref), [`psis_loo`](@ref), [`PsisLoo`](@ref).
"""
function psis(ll_fun::Function, chain::Chains, data::AbstractVector, args...; kwargs...)
    pointwise_log_likes = pointwise_log_likelihoods(ll_fun, chain, data)
    return psis(-pointwise_log_likes, args...; kwargs...)
end
