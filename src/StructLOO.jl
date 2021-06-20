"""
A struct containing the results of Pareto-smoothed improtance sampling. `psis` objects have the following fields:
 - `log_weights`: A vector of smoothed and truncated but *unnormalized* log weights. To get normalized weights use the `weights()` function.
 - `diagnostics`: A named tuple containing two vectors with names `pareto_k` and `n_eff`. 
    - `pareto_k`: Estimates of the shape parameter ``k`` of the generalized Pareto distribution.
    - `n_eff`: Estimated effective sample size for each LOO evaluation.
 - `norm_const_log`: Vector of precomputed values of `colLogSumExps(log_weights)` that are used internally by the `weights` method to normalize the log weights.
 - `tail_len`: Vector of tail lengths used for fitting the generalized Pareto distribution.
 - `r_eff`: If specified, the user's `r_eff` argument.
 - `dims`: Named tuple of length 2 containing `s` (posterior sample size) and `n` (number of observations).
 - `method`: Method used for importance sampling.
"""