export pointwise_log_likelihoods

const ARRAY_DIMS_WARNING = "The supplied array of mcmc samples indicates you have more 
parameters than mcmc samples.This is possible, but highly unusual. Please check that your
array of mcmc samples has the following dimensions: [n_samples,n_parms,n_chains]."

"""
    pointwise_log_likelihoods(
        ll_fun::Function, 
        samples::AbstractArray{<:Real,3}, 
        data;
        splat::Bool=true
    ) 

Compute the pointwise log likelihood.

# Arguments
  - $LIKELIHOOD_FUNCTION_ARG
  - `samples::AbstractArray`: A three dimensional array of MCMC samples. Here, the first
    dimension should indicate the iteration of the MCMC ; the second dimension should
    indicate the parameter ; and the third dimension represents the chains. 
  - `data`: A vector of data used to estimate the parameters of the model.
  - `splat`: If `true` (default), `f` must be a function of `n` different parameters. 
    Otherwise, `f` is assumed to be a function of a single parameter vector.

# Returns
  - `Array`: A three dimensional array of pointwise log-likelihoods.
"""
function pointwise_log_likelihoods(
    ll_fun::Function,
    samples::AbstractArray{<:Union{Real, Missing}, 3},
    data;
    splat::Bool=true,
)
    n_posterior, n_parms, n_chains = size(samples)
    if n_parms > n_posterior
        @info ARRAY_DIMS_WARNING
    end
    if splat
        fun = (p, d) -> ll_fun(p..., d)
    else
        fun = (p, d) -> ll_fun(p, d)
    end
    n_posterior, _, n_chains = size(samples)
    n_data = length(data)
    pointwise_lls = similar(samples, n_data, n_posterior, n_chains)
    for index in CartesianIndices(pointwise_lls)
        datum, iteration, chain = Tuple(index)
        pointwise_lls[datum, iteration, chain] = fun(
            samples[iteration, :, chain], data[datum]
        )
    end
    return pointwise_lls
end


function pointwise_log_likelihoods(
    ll_fun::Function,
    samples::AbstractMatrix{<:Union{Real, Missing}},
    data;
    chain_index::AbstractVector{<:Integer}=_assume_one_chain(samples),
    kwargs...,
)
    samples = _convert_to_array(samples, chain_index)
    return pointwise_log_likelihoods(ll_fun, samples, data)
end

