using .MCMCChains
export pointwise_log_likelihoods

"""
    pointwise_log_likelihoods(chain::Chains, data, ll_fun)

Computes the pointwise log likelihoods where [d,s,c] corresponds to log likelihood of 
evaluated at datapoint d, sample s, for chain c. 

- `chain`: a chain object from MCMCChains
- `data`: a vector of data used to estimate parameters of the model 
- `ll_fun`: a function that computes the log likelihood of a single data point: f(θ1, ..., θn, data), where θi is the ith parameter
"""
function pointwise_log_likelihoods(chain::Chains, data, ll_fun)
    samples = Array(Chains(chain, :parameters).value)
    pointwise_log_likelihoods(samples, data, ll_fun)
end