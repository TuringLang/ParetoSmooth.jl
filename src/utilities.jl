export pointwise_log_likelihoods

"""
    pointwise_log_likelihoods(ll_fun::Function, samples::AbstractArray{<:AstractFloat,3}, data) 

Computes the pointwise log likelihood.

# Arguments
- `ll_fun::Function`: a function that computes the log likelihood of a single data point:
     f(θ1, ..., θn, data), where θi is the ith parameter
- `samples::AbstractArray{<:AstractFloat,3}`: a three dimensional array of mcmc samples 
    where the first dimension represents the samples, the second dimension represents
    the samples, and the third dimension represents the chains. 
- `data`: a vector of data used to estimate parameters of the model 

# Returns
- `Array{Float64,3}`: a three dimensional array of pointwise log likelihoods
"""
function pointwise_log_likelihoods(ll_fun::Function, samples::AbstractArray{<:AbstractFloat,3}, data)
    fun = (p,d)-> ll_fun(p..., d)
    n_data = length(data)
    n_samples, _,n_chains = size(samples)
    pointwise_lls = fill(0.0, n_data, n_samples, n_chains)
    for c in 1:n_chains 
        for s in 1:n_samples
            for d in 1:n_data
                pointwise_lls[d,s,c] = fun(samples[s,:,c], data[d])
            end
        end
    end
    return pointwise_lls
end