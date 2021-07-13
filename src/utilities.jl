using MCMCChains, StatsFuns, Turing
export pointwise_loglikes, compute_loo

"""
    pointwise_loglikes(chain::Chains)

Computes the pointwise log likelihoods where [d,s,c] corresponds to log likelihood of 
evaluated at datapoint d, sample s, for chain c. 

- `chain`: a chain object from MCMCChains from a Turing model
- `model`: a Turing model with data in the form of model(data)
"""
function pointwise_loglikes(chain::Chains, model)
    chain_params = MCMCChains.get_sections(chain, :parameters)
    pwLL_dict = pointwise_loglikelihoods(model, chain_params)
    ind_from_string(x) = parse(Int, x[3:end-1])
    sorted_keys = sort(collect(keys(pwLL_dict)); by=ind_from_string)
    return permutedims(cat((pwLL_dict[k] for k in sorted_keys)...; dims=3), (3, 1, 2))
end

"""
    pointwise_loglikes(chain::Chains, data, ll_fun)

Computes the pointwise log likelihoods where [d,s,c] corresponds to log likelihood of 
evaluated at datapoint d, sample s, for chain c. 

- `chain`: a chain object from MCMCChains
- `data`: a vector of data used to estimate parameters of the model 
- `ll_fun`: a function that computes the log likelihood of a single data point: f(θ1, ..., θn, data), where θi is the ith parameter
"""
function pointwise_loglikes(chain::Chains, data, ll_fun)
    samples = Array(Chains(chain, :parameters).value)
    pointwise_loglikes(samples, data, ll_fun)
end

"""
    pointwise_loglikes(samples::Array{Float64,3}, data, ll_fun)

Computes the pointwise log likelihood 

- `samples`: a three dimensional array of mcmc samples where the first dimension represents the samples, the second dimension represents
the samples, and the third dimension represents the chains. 
- `data`: a vector of data used to estimate parameters of the model 
- `ll_fun`: a function that computes the log likelihood of a single data point: f(θ1, ..., θn, data), where θi is the ith parameter
"""
function pointwise_loglikes(samples::Array{Float64,3}, data, ll_fun)
    fun = (p,d)-> ll_fun(p..., d)
    n_data = length(data)
    n_samples, n_chains = size(samples)[[1,3]]
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

"""
    compute_loo(psis_output, pointwise_lls)

Computes leave one out (loo) cross validation based on posterior distribution of model parameters.

- `psis_output`: object returned from `psis`
- `pointwise_lls`: pointwise log likelihood where `pointwise_lls[d,s,c]` corresponds to log likelihood of 
evaluated at datapoint d, sample s, for chain c.  
"""
function compute_loo(psis_output, pointwise_lls)
    dims = size(pointwise_lls)
    lwp = deepcopy(pointwise_lls)
    lwp += psis_output.weights
    lwpt = reshape(lwp, dims[1], dims[2] * dims[3])'
    loos = reshape(logsumexp(lwpt; dims=1), size(lwpt, 2))
    return sum(loos)
end

"""
    compute_loo(chain::Chain, data, ll_fun)

Computes leave one out (loo) cross validation based on posterior distribution of model parameters.

- `chain`: an MCMCChains object from Turing
- `model`: a Turing model with data in the form of model(data)
"""
function compute_loo(chain::Chains, model)
    pointwise_lls = pointwise_loglikes(chain,  model)
    psis_output = psis(pointwise_lls)
    return compute_loo(psis_output, pointwise_lls)
end

"""
    compute_loo(chain::Chain, data, ll_fun)

Computes leave one out (loo) cross validation based on posterior distribution of model parameters.

- `chain`: a chain object from MCMCChains
- `data`: a vector of data used to estimate parameters of the model 
- `ll_fun`: a function that computes the log likelihood of a single data point: f(θ1, ..., θn, data), where θi is the ith parameter
"""
function compute_loo(chain::Chains, data, ll_fun::Function)
    pointwise_lls = pointwise_loglikes(chain, data, ll_fun)
    psis_output = psis(pointwise_lls)
    return compute_loo(psis_output, pointwise_lls)
end