using AxisKeys
using Bootstrap
using InteractiveUtils
using LoopVectorization
using Memoization
using SpecialFunctions
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
    ess = psis_object.ess
    r_eff = psis_object.r_eff


    @tullio pointwise_ev[i] := weights[i, j, k] * exp(log_likelihood[i, j, k]) |> log
    @tullio pointwise_naive[i] := exp(log_likelihood[i, j, k] - log_count) |> log
    pointwise_mcse = similar(pointwise_ev)
    pointwise_mcse .= mcse_elpd(log_likelihood, weights, pointwise_ev, r_eff)

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
        crit=[:total_score, :overfit, :avg_score],
        est=[:Estimate, :SE],
        )
    
    table(:total_score, :Estimate, :) .= ev_loo = sum(pointwise_ev)
    table(:avg_score, :Estimate, :) .= ev_avg = ev_loo / data_size
    table(:overfit, :Estimate, :) .= p_eff = sum(pointwise_p_eff)

    table(:total_score, :SE, :) .= ev_se = sqrt(varm(pointwise_ev, ev_avg) * data_size)
    table(:avg_score, :SE, :) .= ev_se / data_size
    table(:overfit, :SE, :) .= sqrt(varm(pointwise_p_eff, p_eff / data_size) * data_size)

    return PsisLoo(table, pointwise, psis_object);

end

function elpd(weights, log_likelihood)
    return @tullio pointwise_ev[i] := weights[i, j, k] * exp(log_likelihood[i, j, k]) |> log
end

@memoize function _qnorm(n::Integer)
    correction = 3/8  # results in approximately unbiased sample quantiles
    probs = (1:n .- correction) ./ (n - 2 * correction + 1)
    return erfinv.((1 .+ probs) / 2)
end


function mcse_elpd(log_likelihood, weights, elpd, r_eff, n_samples = 1000)
    likelihood = exp.(log_likelihood)
    exp_elpd = exp.(elpd)  # geometric mean of probability density
    norm_quantiles = _qnorm(n_samples)
    @tullio sd_epd[i] := (weights[i, j, k] * (likelihood[i, j, k] - exp_elpd[i])) ^ 2 |> sqrt
    z = exp_elpd .+ sd_epd .* norm_quantiles'
    var_elpd = dropdims(var(log.(clamp.(z, 0, Inf)); dims=2); dims=2)
    return sqrt.(var_elpd ./ r_eff)
end