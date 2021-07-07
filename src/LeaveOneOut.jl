using AxisKeys
using Distributions
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


function loo(args...; method::AbstractLooMethod=PsisLooMethod(), kwargs...)
    if method ∈ LOO_METHODS
        return psis_loo(args...; kwargs...)
    else
        throw(ArgumentError("Invalid method provided. Valid methods are $LOO_METHODS"))
    end
end


function psis_loo(
    log_likelihood::ArrayType; 
    rel_eff=similar(log_likelihood, 0), 
    boostrap_count::Integer=2^12, 
    conf_int::Tuple{Float64}=two_nat_interval,
) where {F<:AbstractFloat,ArrayType<:AbstractArray{F,3}}


    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
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
    let log_count = log(mcmc_count)
        @tullio pointwise_naive[i] := exp(log_likelihood[i, j, k] - log_count) |> log
    end
    pointwise_p_eff = pointwise_naive - pointwise_ev
    pointwise = StructArray{LooPoint{F}}(
        estimate=pointwise_ev, 
        mcse=pointwise_mcse, 
        p_eff=pointwise_p_eff, 
        pareto_k=ξ
        )

    table = KeyedArray(similar(log_likelihood, (3,2)); 
        estimate=[:Estimate, :SE], criterion=[:loo, :p_loo, :loo_ic]
        )
    
    # Use Bayesian bootstrap to build confidence intervals
    table[criterion=:loo, estimate=:Estimate] = ev_loo = mean(pointwise_ev)
    table[criterion=:p_loo, estimate=:Estimate] = p_eff = sum(pointwise_p_eff)
    table[criterion=:loo_ic, estimate=:Estimate] = -2 * ev_loo

    ev_loo_se = sqrt(var(pointwise_ev; mean=ev_loo) / data_size)
    p_eff_se = sqrt(var(pointwise_p_eff; mean=p_eff/data_size) * data_size)


    return PsisLoo(table, pointwise, psis_object)

end
