using AxisKeys
using InteractiveUtils
using LoopVectorization
using ParetoSmooth
using Statistics
using StructArrays
using Tullio

export loo, psis_loo

const LOO_METHODS = subtypes(AbstractLooMethod)
const TWO_NAT_PROB = exp(-2) / (exp(-2) + 1)
const TWO_NAT_INTERVAL = (TWO_NAT_PROB / 2, 1 - TWO_NAT_PROB/2)


function loo(args...; method=PsisLooMethod(), kwargs...)
    if typeof(method) ∈ LOO_METHODS
        return psis_loo(args...; kwargs...)
    else
        throw(ArgumentError("Invalid method provided. Valid methods are $LOO_METHODS"))
    end
end


function psis_loo(log_likelihood::ArrayType; 
    rel_eff=similar(log_likelihood, 0), 
    conf_int::Tuple=TWO_NAT_INTERVAL,
) where {F<:AbstractFloat,ArrayType<:AbstractArray{F,3}}


    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
    log_count = log(mcmc_count)
    # TODO: Add a way of using score functions other than ELPD
    # log_likelihood::ArrayType = similar(log_likelihood)
    # log_likelihood .= score(log_likelihood)

    psis_object = psis(-log_likelihood, rel_eff)
    weights = psis_object.weights
    ξ = psis_object.pareto_k
    ess = psis_object.ess

    @tullio pointwise_ev[i] := weights[i, j, k] * exp(log_likelihood[i, j, k]) |> log
    # Replace with quantile mcse estimate from R LOO package?
    @tullio pointwise_mcse[i] := sqrt <|
        (weights[i, j, k] * log_likelihood[i, j, k] - pointwise_ev[i])^2 / ess[i]
    @tullio pointwise_naive[i] := exp(log_likelihood[i, j, k] - log_count) |> log
    
    pointwise_p_eff = pointwise_naive - pointwise_ev
    pointwise = KeyedArray(hcat(
                    pointwise_ev,
                    pointwise_mcse,
                    pointwise_p_eff,
                    ξ,
                );
                data=1:length(pointwise_ev),
                statistic=[:est_score, :mcse_score, :est_overfit, :pareto_k]
    )
                
    

    table = KeyedArray(similar(log_likelihood, 2, 2); 
        crit=[:loo, :p_loo],
        est=[:Estimate, :SE],
        )
    
    # Use Bayesian bootstrap to build confidence intervals
    table[Key(:loo), Key(:Estimate)] = ev_loo = mean(pointwise_ev)
    table[Key(:p_loo), Key(:Estimate)] = p_eff = sum(pointwise_p_eff)

    table[Key(:loo), Key(:SE)] = se = sqrt(varm(pointwise_ev, ev_loo) / data_size)
    table[Key(:p_loo), Key(:SE)] = sqrt(varm(pointwise_p_eff, p_eff / data_size) * data_size)

    return PsisLoo(table, pointwise, psis_object);

end
