using LoopVectorization
using ParetoSmooth
using Statistics
using StructArrays
using Tullio

export loo, psis_loo, PsisLoo, Loo, AbstractLoo, LooMethod, PsisLooMethod


struct LooPoint{F<:AbstractFloat}
    estimate::F
    mcse::F
    p_eff::F
    pareto_k::F
end


abstract type AbstractLoo end


struct PsisLoo{
    F<:AbstractFloat,
    AF<:AbstractArray{F},
    VF<:AbstractVector{F},
    I<:Integer,
    VI<:AbstractVector{I},
} <: AbstractLoo
    estimates::Dict{String,F}
    pointwise::StructArray{LooPoint{F}}
    psis_object::Psis{F,AF,VF,I,VI}
end


abstract type LooMethod end

struct PsisLooMethod <: LooMethod end

const LOO_METHODS = [PsisLooMethod(),]


function loo(args...; method::LooMethod=PsisLooMethod(), kwargs...)
    if method ∈ LOO_METHODS
        return psis_loo(args...; kwargs...)
    else
        throw(ArgumentError("Invalid method provided. Valid methods are $LOO_METHODS"))
    end
end


function psis_loo(
    log_likelihood::ArrayType; rel_eff=similar(log_likelihood, 0)
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
    @tullio pointwise_naive[i] := exp(log_likelihood[i, j, k] - log(mcmc_count)) |> log
    pointwise_p_eff = pointwise_naive - pointwise_ev
    points = (estimate=pointwise_ev, mcse=pointwise_mcse, p_eff=pointwise_p_eff, pareto_k=ξ)
    pointwise = StructArray{LooPoint{F}}(points)


    @tullio ev := points.estimate[i]
    @tullio ev_naive := pointwise_naive[i]
    p_eff = ev_naive - ev

    ev_se = sqrt(var(pointwise_ev; mean=ev) / data_size)
    p_eff_se = sqrt(var(pointwise_p_eff; mean=p_eff / data_size) / data_size)

    vals = Dict(
        "Score Est" => ev,
        "Parameters (Eff)" => p_eff,
        "SE(Score Est)" => ev_se,
        "SE(Parameters)" => p_eff_se,
    )

    return PsisLoo(vals, pointwise, psis_object)

end

