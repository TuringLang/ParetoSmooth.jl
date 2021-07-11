using AxisKeys
using Bootstrap
using InteractiveUtils
using LoopVectorization
using Statistics
using Tullio


export loo, psis_loo

const LOO_METHODS = subtypes(AbstractLooMethod)

function loo(args...; method=PsisLooMethod(), kwargs...)
    if typeof(method) ∈ LOO_METHODS
        return psis_loo(args...; kwargs...)
    else
        throw(ArgumentError("Invalid method provided. Valid methods are $LOO_METHODS"))
    end
end


function psis_loo(log_likelihood::T; 
    r_eff=similar(log_likelihood, 0),
    source::Union{AbstractString,Symbol}="mcmc", 
    log_weights::Bool=false
) where {F<:AbstractFloat,T<:AbstractArray{F,3}}

    source = lowercase(string(source))
    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
    log_count = log(mcmc_count)


    # TODO: Add a way of using score functions other than ELPD
    # log_likelihood::ArrayType = similar(log_likelihood)
    # log_likelihood .= score(log_likelihood)

    psis_object = psis(-log_likelihood, r_eff; source=source, log_weights=log_weights)
    weights = psis_object.weights
    ξ = psis_object.pareto_k
    r_eff = psis_object.r_eff


    @tullio pointwise_ev[i] := weights[i, j, k] * exp(log_likelihood[i, j, k]) |> log
    @tullio pointwise_naive[i] := exp(log_likelihood[i, j, k] - log_count) |> log
    @tullio pointwise_mcse[i] := (weights[i, j, k] * (log_likelihood[i, j, k] - pointwise_ev[i]))^2
    @. pointwise_mcse = sqrt(pointwise_mcse / r_eff)
    

    pointwise_p_eff = pointwise_naive - pointwise_ev
    pointwise = KeyedArray(hcat(
                    pointwise_ev,
                    pointwise_mcse,
                    pointwise_p_eff,
                    ξ,
                );
                data=1:length(pointwise_ev),
                statistic=[
                    :est_score, 
                    :mcse, 
                    :est_overfit, 
                    :pareto_k
                ]
    )
                
    table = KeyedArray(similar(log_likelihood, 3, 2); 
        criterion=[:total_score, :overfit, :avg_score],
        estimate=[:Estimate, :SE],
        )
    
    table(:total_score, :Estimate, :) .= ev_loo = sum(pointwise_ev)
    table(:avg_score, :Estimate, :) .= ev_avg = ev_loo / data_size
    table(:overfit, :Estimate, :) .= p_eff = sum(pointwise_p_eff)

    table(:total_score, :SE, :) .= ev_se = sqrt(varm(pointwise_ev, ev_avg) * data_size)
    table(:avg_score, :SE, :) .= ev_se / data_size
    table(:overfit, :SE, :) .= sqrt(varm(pointwise_p_eff, p_eff / data_size) * data_size)

    return PsisLoo(table, pointwise, psis_object);

end

