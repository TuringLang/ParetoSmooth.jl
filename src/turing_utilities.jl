using Turing
"""
    compute_loo(chain::Chain, data, ll_fun)

Computes leave one out (loo) cross validation based on posterior distribution of model parameters.

- `chain`: an MCMCChains object from Turing
- `model`: a Turing model with data in the form of model(data)
"""
function compute_loo(chain::Chains, model)
    pointwise_lls = pointwise_loglikes(chain, model)
    psis_output = psis(pointwise_lls)
    return compute_loo(psis_output, pointwise_lls)
end

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
