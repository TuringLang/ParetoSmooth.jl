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
    psis_loo(ll_fun::Function, chain::Chains, data;
    source::Union{AbstractString,Symbol}="mcmc", 
    log_weights::Bool=false
    )

    # Arguments
- `ll_fun::Function`: a function that computes the log likelihood of a single data point: 
    f(θ1, ..., θn, data), where θi is the ith parameter
- `chain::Chain`: a chain object from MCMCChains
- `data`: a vector of data used to estimate parameters of the model

# Returns
- `Array{Float64,3}`: a three dimensional array of pointwise log likelihoods 
"""
function psis_loo(ll_fun::Function, 
    chain::Chains, 
    data;
    source::Union{AbstractString,Symbol}="mcmc", 
    log_weights::Bool=false
    )
    pointwise_log_likes = pointwise_log_likelihoods(ll_fun, chain, data)
    r_eff = similar(pointwise_log_likes, 0)
    return psis_loo(pointwise_log_likes, r_eff; source, log_weights)
end

