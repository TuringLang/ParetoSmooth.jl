module PSIS

using Base: AbstractFloat
using LoopVectorization
using Tullio

include("GPD.jl")
include("ESS.jl")
include("LooUtility.jl")

export psis

const SAMPLE_SOURCES = [:mcmc, :vi, :other]


"""
    psis(log_ratios::AbstractArray{T:>AbstractFloat}; source::Union{AbstractString, Symbol}="mcmc", r_call::Bool=false)
Implements Pareto-smoothed importance sampling.

# Arguments
- `log_ratios::AbstractArray`: An array of importance ratios on the log scale (for PSIS-LOO these are *negative* log-likelihood values). Indices must be `[data, draw, chain]` -- `log_ratios[1, 2, 3]` should be the log-likelihood of the first data point, evaluated at the second second iteration of the third chain. Chain indices can be left off if there is only a single chain.
- `source::Union{AbstractString, Symbol}="mcmc"`: A string or symbol describing the source of the sample being used. If `"mcmc"`, adjusts ESS for autocorrelation. Otherwise, samples are assumed to be independent. Currently permitted values are $SAMPLE_SOURCES.
- `r_call::Bool=false`: should be used only if called from R.
"""
function psis(log_ratios::T; source::Union{AbstractString, Symbol}="mcmc", r_call::Bool=false) where {F<:AbstractFloat, T<:AbstractArray{F, 3}}
    
    source = lowercase(String(source))
    dimensions = size(log_ratios)
    
    numDataPoints = dimensions[1]
    posteriorSampleSize = dimensions[2] * dimensions[3]
    

    # Reshape to matrix (easier to deal with)
    log_ratios = reshape(log_ratios, numDataPoints, posteriorSampleSize)
    weights = similar(log_ratios, numDataPoints, posteriorSampleSize)
    @tturbo weights .= exp.(log_ratios .- maximum(log_ratios; dims=2))

    if source == "mcmc"
        @info "Adjusting for autocorrelation. If the posterior samples are not autocorrelated, specify the source of the posterior sample using the keyword argument `source`. MCMC samples are always autocorrelated."
        relEff = ESS.relative_eff(reshape(weights, dimensions))
    end

    tailLength = similar(log_ratios, Int, numDataPoints)
    ξ = similar(log_ratios, F, numDataPoints)
    for i in 1:numDataPoints
        tailLength[i] = def_tail_length(posteriorSampleSize, relEff[i])
        ξ[i] = do_psis_i!(view(weights, i, :), tailLength[i])
    end

    @tullio normConst[i] := weights[i, j]
    
    if ~r_call
        @tturbo @. weights = weights / normConst
    end
    
    if source == "mcmc"
        ess = ESS.psis_n_eff(weights, relEff)
    else
        ess = ESS.psis_n_eff(weights)
    end

    weights = reshape(weights, dimensions)

    if r_call
        @tturbo @. weights = log(weights)
    end

    return Psis(
        weights,
        ξ, 
        ess,
        tailLength,
        relEff,
        posteriorSampleSize,
        numDataPoints,
        "psis"
    )
    
end


function psis(log_ratios::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat}
    @info "Chain information was not provided; all samples are assumed to be drawn from a single chain."
    return psis(reshape(log_ratios, size(log_ratios), 1); kwargs...)
end


"""
    do_psis_i!(is_ratios::AbstractVector{T <: AbstractFloat}, tail_length::Integer)
Do PSIS on a single vector, smoothing its tail values.

# Arguments
- `is_ratios::AbstractVector{T<:AbstractFloat}`: A vector of (not necessarily normalized) importance sampling ratios.

# Returns
- `T<:AbstractFloat`: ξ, the shape parameter for the GPD; larger numbers indicate thicker tails.
- `T<:AbstractFloat`: The effective sample size.
- `AbstractVector{T<:AbstractFloat}`: A vector of smoothed importance sampling ratios.

# Extended help
Additional information can be found in the LOO package from R {Add link here}
{Add citation for Vehtari paper}
"""
function do_psis_i!(
    is_ratios::AbstractVector{T},
    tail_length::Integer
) where {T<:AbstractFloat}
    len = length(is_ratios)
    # Sort a vector in place, returning the indices that would be returned by "sortperm"
    ordering = sortperm(is_ratios; alg=QuickSort)
    permute!(is_ratios, ordering)

    # Define and check tail
    tailStartsAt = len - tail_length + 1  # index of smallest tail value
    @views tail = is_ratios[tailStartsAt:len]
    LooUtility.check_tail(tail)
    
    # Get value just before the tail starts:
    cutoff = is_ratios[tailStartsAt - 1]
    ξ = psis_smooth_tail!(tail, cutoff)

    # truncate at max of raw wts (i.e. 1 since largest weight is divided out)
    clamp!(is_ratios, -Inf, 1)
    # unsort the ratios to their original position:
    permute!(is_ratios, collect(1:len)[ordering])

    return ξ::T
end


"""
    def_tail_length(log_ratios::AbstractVector, r_eff::AbstractFloat)
Define the tail length as in Vehtari et al. (2019).
"""
@noinline function def_tail_length(length::Integer, r_eff::AbstractFloat)
    ess = Int(length ÷ r_eff)
    return min(ess ÷ 5, 3 * isqrt(ess)) + 1
end



"""
    psis_smooth_tail!(tail::AbstractVector)
Takes an *already sorted* vector of observations from the tail and smooths it in place with PSIS before returning shape parameter `ξ`.
"""
function psis_smooth_tail!(
    tail::AbstractVector{T},
    cutoff::Real,
) where {T<:AbstractFloat}
    len = length(tail)
    @turbo @. tail = tail - cutoff

    # save time not sorting since x already sorted
    ξ, σ = @turbo GPD.gpdfit(tail)
    if ξ != Inf
        @turbo @. tail = GPD.gpd_quantile($(1:len) / (len + 1), ξ, σ) + cutoff
    end
    return ξ
end


"""
    Psis{V<:AbstractVector{F},I<:Integer} where {F<:AbstractFloat}
A struct containing the results of Pareto-smoothed improtance sampling. `Psis` objects have the following fields:
- `log_weights`: A vector of smoothed and truncated but *unnormalized* log weights. To get normalized weights use the `weights()` function.
- `diagnostics`: A named tuple containing two vectors with names `pareto_k` and `n_eff`. 
    - `pareto_k`: Estimates of the shape parameter ``k`` of the generalized Pareto distribution.
    - `n_eff`: Estimated effective sample size for each LOO evaluation.
- `tail_len`: Vector of tail lengths used for fitting the generalized Pareto distribution.
- `r_eff`: If specified, the user's `r_eff` argument.
- `dims`: Named tuple of length 2 containing `s` (posterior sample size) and `n` (number of observations).
- `method`: String describing the method used for importance sampling.
"""
struct Psis{F<:AbstractFloat, AF<: AbstractArray{F}, VF<:AbstractVector{F}, I<:Integer, VI<:AbstractVector{I}}
    weights::AF
    pareto_k::VF
    ess::VF
    tail_len::VI
    r_eff::VF
    posterior_sample_size::I
    data_size::I
    method::String
end

end
