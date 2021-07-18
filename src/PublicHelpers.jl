export pointwise_log_likelihoods

"""
    pointwise_log_likelihoods(
        ll_fun::Function, 
        samples::AbstractArray{<:AstractFloat,3}, 
        data;
        splat::Bool=true
    ) 

Compute the pointwise log likelihood.

# Arguments
  - $LIKELIHOOD_FUNCTION_ARG
  - `samples::AbstractArray`: A three dimensional array of MCMC samples. Here, the first
    dimension should indicate the parameter being sampled; the second dimension should
    indicate the iteration of the MCMC ; and the third dimension represents the chains. 
  - `data`: A vector of data used to estimate the parameters of the model.
  - `splat`: If `true` (default), `f` must be a function of `n` different parameters. 
    Otherwise, `f` is assumed to be a function of a single parameter vector.

# Returns
  - `Array`: a three dimensional array of pointwise log-likelihoods.
"""
function pointwise_log_likelihoods(
    ll_fun::Function, 
    samples::AbstractArray{<:AbstractFloat, 3}, 
    data;
    splat::Bool=true
)
    if splat 
        fun = (p, d) -> ll_fun(p..., d)
    else
        fun = (p, d) -> ll_fun(p, d)
    end
    _, n_posterior, n_chains = size(samples)
    n_data = length(data)  # First index will represent data, not parameters.
    pointwise_lls = similar(samples, n_data, n_posterior, n_chains)
    for index in CartesianIndices(pointwise_lls)
        datum, iteration, chain = Tuple(index)
        pointwise_lls[datum, iteration, chain] = 
            fun(samples[:, iteration, chain], data[datum])
    end
    return pointwise_lls
end


function pointwise_log_likelihoods(
    ll_fun::Function, 
    samples::AbstractMatrix{<:AbstractFloat}, 
    data;
    chain_index::AbstractVector{<:Integer}=_assume_one_chain(samples),
    kwargs...
)
    samples = _convert_to_array(samples, chain_index)
    return pointwise_log_likelihoods(ll_fun, samples, data; kwargs...)
end

