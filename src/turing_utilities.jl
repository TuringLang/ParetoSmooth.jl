using .Turing, .MCMCChains
export pointwise_log_likelihoods

"""
    pointwise_log_likelihoods(chain::Chains, model)

Computes the pointwise log likelihoods from Turing model where [d,s,c] corresponds to log likelihood of 
evaluated at datapoint d, sample s, for chain c. Note that currently the posterior log likelihood must be 
computed in a for loop in the Turing model.

- `chain`: a chain object from MCMCChains from a Turing model
- `model`: a Turing model with data in the form of model(data)
"""
function pointwise_log_likelihoods(chain::Chains, model)
    # subset of chain for mcmc samples
    chain_params = MCMCChains.get_sections(chain, :parameters)
    # compute the pointwise log likelihoods
    pointwise_log_like_dict = pointwise_loglikelihoods(model, chain_params)
    ind_from_string(x) = parse(Int, x[3:end-1])
    # collect variable names
    sorted_keys = sort(collect(keys(pointwise_log_like_dict)); by=ind_from_string)
    # Convert from dictionary to 3d array
    return permutedims(cat((pointwise_log_like_dict[k] for k in sorted_keys)...; dims=3), (3, 1, 2))
end
