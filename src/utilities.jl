using Tullio
export pointwise_log_likelihoods

"""
    pointwise_log_likelihoods(samples::Array{<:AbstractFloat,3}, data, ll_fun)

Computes the pointwise log likelihood 

- `samples`: a three dimensional array of mcmc samples where the first dimension represents the samples, the second dimension represents
the samples, and the third dimension represents the chains. 
- `data`: a vector of data used to estimate parameters of the model 
- `ll_fun`: a function that computes the log likelihood of a single data point: f(θ1, ..., θn, data), where θi is the ith parameter
"""
function pointwise_log_likelihoods(samples::Array{<:AbstractFloat,3}, data, ll_fun)
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

# the code in the function works, but does not work when called from the function
# function pointwise_log_likelihoods(samples::Array{<:AbstractFloat,3}, data, ll_fun)
#     fun = (p,d)-> ll_fun(p..., d)
#     @tullio pointwise_lls[d,s,c] := fun(samples[s,:,c], data[d])
# end

# # the code in the function works, but does not work when called from the function
# function pointwise_log_likelihoods(samples::Array{<:AbstractFloat,3}, data, ll_fun)
#     println("hi")
#     x = rand(3, 3, 3)
#     y = rand(5)
#     f(x, y) = sum(x) / y
#     @tullio m[d,s,c] := f(x[s,:,c], y[d])
# end