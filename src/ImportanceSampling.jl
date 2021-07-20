using LoopVectorization
using Polyester
using SortingLab
using Tullio


const LIKELY_ERROR_CAUSES = """
1. Bugs in the program that generated the sample, or otherwise incorrect input variables. 
2. Your chains failed to converge. Check diagnostics. 
3. You do not have enough posterior samples (ESS < ~25).
"""
const MIN_TAIL_LEN = 5  # Minimum size of a tail for PSIS to give sensible answers
const SAMPLE_SOURCES = ["mcmc", "vi", "other"]

export Psis, psis

"""
    psis(
        log_ratios::AbstractArray{T:>AbstractFloat}, 
        r_eff::AbstractVector; 
        source::String="mcmc", 
        log_weights::Bool=false
    ) -> Psis

Implements Pareto-smoothed importance sampling (PSIS).

# Arguments
## Positional Arguments
  - `log_ratios::AbstractArray`: A 2d or 3d array of importance ratios on the log scale (for 
    PSIS-LOO these are *negative* log-likelihood values). Indices must be ordered as 
    `[data, step, chain]`: `log_ratios[1, 2, 3]` should be the log-likelihood of the first 
    data point, evaluated at the second step in the third chain. Chain indices can be
    left off if there is only one chain, or if keyword argument `chain_index` is provided.
  - $R_EFF_DOC

## Keyword Arguments

  - $CHAIN_INDEX_DOC
  - `source::String="mcmc"`: A string or symbol describing the source of the sample being 
    used. If `"mcmc"`, adjusts ESS for autocorrelation. Otherwise, samples are assumed to be 
    independent. Currently permitted values are $SAMPLE_SOURCES.
  - `log_weights::Bool=false`: Return the log weights, rather than the PSIS weights. 

See also: [`relative_eff`]@ref, [`psis_loo`]@ref, [`psis_ess`]@ref.
"""
function psis(
    log_ratios::AbstractArray{T, 3},
    r_eff::AbstractVector{T}=similar(log_ratios, 0);
    source::Union{AbstractString,Symbol}="mcmc",
    log_weights::Bool=false,
) where {T<:AbstractFloat}

    source = lowercase(String(source))
    dims = size(log_ratios)

    data_size = dims[1]
    post_sample_size = dims[2] * dims[3]

    # Reshape to matrix (easier to deal with)
    log_ratios = reshape(log_ratios, data_size, post_sample_size)
    weights::AbstractArray{T} = similar(log_ratios)
    # Shift ratios by maximum to prevent overflow
    @tturbo @. weights = exp(log_ratios - $maximum(log_ratios; dims=2))

    r_eff = _generate_r_eff(weights, dims, r_eff, source)
    _check_input_validity_psis(reshape(log_ratios, dims), r_eff)

    tail_length = similar(log_ratios, Int, data_size)
    ξ = similar(log_ratios, data_size)
    Threads.@threads for i in eachindex(tail_length)
        tail_length[i] = _def_tail_length(post_sample_size, r_eff[i])
        ξ[i] = @views ParetoSmooth._do_psis_i!(weights[i,:], tail_length[i])
    end
    
    @tullio norm_const[i] := weights[i, j]
    @turbo weights .= weights ./ norm_const
    ess = psis_ess(weights, r_eff)

    weights = reshape(weights, dims)

    if log_weights
        @tturbo @. weights = log(weights)
    end

    return Psis(weights, ξ, ess, r_eff, tail_length, post_sample_size, data_size)
end


function psis(
    log_ratios::AbstractMatrix{T},
    r_eff::AbstractVector{T}=similar(log_ratios, 0);
    chain_index::AbstractVector{I}=_assume_one_chain(log_ratios),
    kwargs...,
) where {T<:AbstractFloat,I<:Integer}
    
    new_log_ratios = _convert_to_array(log_ratios, chain_index)
    return psis(new_log_ratios, r_eff; kwargs...)
end


"""
    _do_psis_i!(is_ratios::AbstractVector{AbstractFloat}, tail_length::Integer) -> T

Do PSIS on a single vector, smoothing its tail values.

# Arguments

- `is_ratios::AbstractVector{AbstractFloat}`: A vector of importance sampling ratios, 
scaled to have a maximum of 1.

# Returns

- `T<:AbstractFloat`: ξ, the shape parameter for the GPD; big numbers indicate thick tails.

# Extended help

Additional information can be found in the LOO package from R.
"""
function _do_psis_i!(
    is_ratios::AbstractVector{T}, tail_length::Integer
) where {T<:AbstractFloat}
    len = length(is_ratios)
    
    # sort is_ratios and also get results of sortperm() at the same time
    ordering = fsortperm(is_ratios)
    sorted_ratios = is_ratios[ordering]

    # Define and check tail
    tail_start = len - tail_length + 1  # index of smallest tail value
    @views tail = sorted_ratios[tail_start:len]
    _check_tail(tail)

    # Get value just before the tail starts:
    cutoff = sorted_ratios[tail_start - 1]
    ξ = _psis_smooth_tail!(tail, cutoff)

    # truncate at max of raw weights (1 after scaling)
    clamp!(sorted_ratios, 0, 1)
    # unsort the ratios to their original position:
    is_ratios .= @views sorted_ratios[invperm(ordering)]

    return ξ::T
end


"""
    _def_tail_length(log_ratios::AbstractVector, r_eff::AbstractFloat) -> tail_len::Integer

Define the tail length as in Vehtari et al. (2019).
"""
function _def_tail_length(length::I, r_eff::AbstractFloat) where {I<:Integer}
    len = I(ceil(min(length / 5, 3 * sqrt(length / r_eff))))
    len = 4 * round(len / 4) # multiples of 4 easier to vectorize
    return I(len)
end


"""
    _psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T) where {T<:AbstractFloat} -> ξ::T

Takes an *already sorted* vector of observations from the tail and smooths it *in place*  
with PSIS before returning shape parameter `ξ`.
"""
function _psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T) where {T<:AbstractFloat}
    len = length(tail)
    @turbo @. tail = tail - cutoff

    # save time not sorting since tail is already sorted
    ξ, σ = gpdfit(tail)
    if ξ ≠ Inf
        @turbo @. tail = gpd_quantile(($(1:len) - 0.5) / len, ξ, σ) + cutoff
    end
    return ξ
end



##########################
#### HELPER FUNCTIONS ####
##########################

"""
Generate the relative effective sample size if not provided by the user.
"""
function _generate_r_eff(weights::AbstractArray, dims, r_eff::AbstractArray, source::String)
    if isempty(r_eff)
        if source == "mcmc"
            @info "Adjusting for autocorrelation. If the posterior samples are not " *
                  "autocorrelated, specify the source of the posterior sample using the " *
                  "keyword argument `source`. MCMC samples are always autocorrelated; VI " *
                  "samples are not."
            return relative_eff(reshape(weights, dims))
        elseif source ∈ SAMPLE_SOURCES
            @info "Samples have not been adjusted for autocorrelation. If the posterior " *
                  "samples are autocorrelated, as in MCMC methods, ESS estimates will be " *
                  "upward-biased, and standard error estimates will be downward-biased. " *
                  "MCMC samples are always autocorrelated; VI samples are not."
            return ones(size(weights, 1))
        else
            throw(
                ArgumentError(
                    "$source is not a valid source. Valid sources are $SAMPLE_SOURCES."
                ),
            )
            return ones(size(weights, 1))
        end
    else
        return r_eff
    end
end

    
"""
Make sure all inputs to `psis` are valid.
"""
function _check_input_validity_psis(
    log_ratios::AbstractArray{T,3}, r_eff::AbstractVector{T}
) where {T<:AbstractFloat}
    if any(isnan, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains NaN values)."))
    elseif any(isinf, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains infinite values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `log_ratios` (array is empty)."))
    elseif any(isnan, r_eff)
        throw(ArgumentError("Invalid input for `r_eff` (contains NaN values)."))
    elseif any(isinf, r_eff)
        throw(DomainError("Invalid input for `r_eff` (contains infinite values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `r_eff` (array is empty)."))
    elseif length(r_eff) ≠ size(log_ratios, 1)
        throw(ArgumentError("Size of `r_eff` does not equal the number of data points."))
    end
    return nothing
end


"""
Check the tail to make sure a GPD fit is possible.
"""
function _check_tail(tail::AbstractVector{T}) where {T<:AbstractFloat}
    if maximum(tail) ≈ minimum(tail)
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution: all tail values are the" *
                "same. Likely causes are: \n$LIKELY_ERROR_CAUSES",
            ),
        )
    elseif length(tail) < MIN_TAIL_LEN
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution: tail length was too " *
                "short. Likely causese are: \n$LIKELY_ERROR_CAUSES"
            ),
        )
    end
    return nothing
end



##########################
#####  PSIS STRUCTS  #####
##########################

"""
    Psis{V<:AbstractVector{F},I<:Integer} where {F<:AbstractFloat}

A struct containing the results of Pareto-smoothed importance sampling.

# Fields
  - `weights`: A vector of smoothed, truncated, and normalized importance sampling weights.
  - `pareto_k`: Estimates of the shape parameter `k` of the generalized Pareto distribution.
  - `ess`: Estimated effective sample size for each LOO evaluation.
  - `tail_len`: Vector indicating how large the "tail" is for each observation.
  - `dims`: Named tuple of length 2 containing `s` (posterior sample size) and `n` (number
    of observations).
"""
struct Psis{
    F<:AbstractFloat,
    AF<:AbstractArray{F,3},
    VF<:AbstractVector{F},
    I<:Integer,
    VI<:AbstractVector{I},
}
    weights::AF
    pareto_k::VF
    ess::VF
    r_eff::VF
    tail_len::VI
    posterior_sample_size::I
    data_size::I
end
