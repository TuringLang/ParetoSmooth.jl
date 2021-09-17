using LoopVectorization
using StatsBase
using Tullio

const LIKELY_ERROR_CAUSES = """
1. Incorrect inputs -- check your program for bugs. If you provided an `r_eff` argument,
double check it is correct.
2. Your chains failed to converge. Check diagnostics. 
3. You do not have enough posterior samples (ESS < ~25).
"""
const MIN_TAIL_LEN = 5  # Minimum size of a tail for PSIS to give sensible answers
const SAMPLE_SOURCES = ["mcmc", "vi", "other"]

export psis, PsisLoo, PsisLooMethod, Psis


###########################
###### RESULT STRUCT ######
###########################


"""
    Psis{R<:Real, AT<:AbstractArray{R, 3}, VT<:AbstractVector{R}}

A struct containing the results of Pareto-smoothed importance sampling.

# Fields

  - `weights`: A vector of smoothed, truncated, and normalized importance sampling weights.
  - `pareto_k`: Estimates of the shape parameter `k` of the generalized Pareto distribution.
  - `ess`: Estimated effective sample size for each LOO evaluation, based on the variance of
    the weights.
  - `sup_ess`: Estimated effective sample size for each LOO evaluation, based on the 
    supremum norm, i.e. the size of the largest weight. More likely than `ess` to warn when 
    importance sampling has failed. However, it can have a high variance.
  - `r_eff`: The relative efficiency of the MCMC chain, i.e. ESS / posterior sample size.
  - `tail_len`: Vector indicating how large the "tail" is for each observation.
  - `posterior_sample_size`: How many draws from an MCMC chain were used for PSIS.
  - `data_size`: How many data points were used for PSIS.
"""
struct Psis{
    RealType <: Real,
    AT <: AbstractArray{RealType, 3},
    VT <: AbstractVector{RealType},
}
    weights::AT
    pareto_k::VT
    ess::VT
    sup_ess::VT
    r_eff::VT
    tail_len::Vector{Int}
    posterior_sample_size::Int
    data_size::Int
end


function Base.show(io::IO, ::MIME"text/plain", psis_object::Psis)
    table = hcat(psis_object.pareto_k, psis_object.ess, psis_object.sup_ess)
    post_samples = psis_object.posterior_sample_size
    data_size = psis_object.data_size
    println("Results of PSIS with $post_samples posterior samples and $data_size cases.")
    _throw_pareto_k_warning(psis_object.pareto_k)
    return pretty_table(
        table;
        compact_printing=false,
        header=[:pareto_k, :ess, :sup_ess],
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end



###########################
####### PSIS FNCTNS #######
###########################


"""
    psis(
        log_ratios::AbstractArray{T<:Real}, 
        r_eff::AbstractVector; 
        source::String="mcmc"    
    ) -> Psis

Implements Pareto-smoothed importance sampling (PSIS).

# Arguments
## Positional Arguments
  - `log_ratios::AbstractArray`: A 2d or 3d array of (unnormalized) importance ratios on the
    log scale. Indices must be ordered as `[data, step, chain]`. The chain index can be left 
    off if there is only one chain, or if keyword argument `chain_index` is provided.
  - $R_EFF_DOC

## Keyword Arguments

  - $CHAIN_INDEX_DOC
  - `source::String="mcmc"`: A string or symbol describing the source of the sample being 
    used. If `"mcmc"`, adjusts ESS for autocorrelation. Otherwise, samples are assumed to be 
    independent. Currently permitted values are $SAMPLE_SOURCES.

See also: [`relative_eff`]@ref, [`psis_loo`]@ref, [`psis_ess`]@ref.
"""
function psis(
    log_ratios::AbstractArray{<:Real, 3};
    r_eff::AbstractVector{<:Real}=similar(log_ratios, 0),
    source::Union{AbstractString, Symbol}="mcmc"
)

    source = lowercase(String(source))
    dims = size(log_ratios)

    data_size = dims[1]
    post_sample_size = dims[2] * dims[3]

    # Reshape to matrix (easier to deal with)
    log_ratios = reshape(log_ratios, data_size, post_sample_size)
    r_eff = _generate_r_eff(log_ratios, dims, r_eff, source)
    weights = similar(log_ratios)
    # Shift ratios by maximum to prevent overflow
    @tturbo @. weights = exp(log_ratios - $maximum(log_ratios; dims=2))
    
    _check_input_validity_psis(reshape(log_ratios, dims), r_eff)

    tail_length = Vector{Int}(undef, data_size)
    ξ = similar(r_eff)
    @inbounds Threads.@threads for i in eachindex(tail_length)
        tail_length[i] = _def_tail_length(post_sample_size, r_eff[i])
        ξ[i] = @views ParetoSmooth._do_psis_i!(weights[i, :], tail_length[i])
    end

    @tullio norm_const[i] := weights[i, j]
    @tturbo weights .= weights ./ norm_const
    ess = psis_ess(weights, r_eff)
    inf_ess = sup_ess(weights, r_eff)

    weights = reshape(weights, dims)

    return Psis(
        weights, 
        ξ, 
        ess, 
        inf_ess, 
        r_eff, 
        tail_length, 
        post_sample_size, 
        data_size
    )
end


function psis(
    log_ratios::AbstractMatrix{<:Real};
    chain_index::AbstractVector{<:Integer}=_assume_one_chain(log_ratios),
    kwargs...,
)
    new_log_ratios = _convert_to_array(log_ratios, chain_index)
    return psis(new_log_ratios; kwargs...)
end


"""
    _do_psis_i!(is_ratios::AbstractVector{Real}, tail_length::Integer) -> T

Do PSIS on a single vector, smoothing its tail values.

# Arguments

  - `is_ratios::AbstractVector{Real}`: A vector of importance sampling ratios,
    scaled to have a maximum of 1.

# Returns

  - `T<:Real`: ξ, the shape parameter for the GPD; big numbers indicate thick tails.
"""
function _do_psis_i!(is_ratios::AbstractVector{T}, tail_length::Integer) where {T <: Real}

    len = length(is_ratios)
    tail_start = len - tail_length + 1  # index of smallest tail value

    # sort is_ratios and also get results of sortperm() at the same time
    ratio_index = collect(zip(is_ratios, Base.OneTo(len)))
    partialsort!(ratio_index, (tail_start-1):len; by=first)
    is_ratios .= first.(ratio_index)
    @views tail = is_ratios[tail_start:len]
    _check_tail(tail)

    # Get value just before the tail starts:
    cutoff = is_ratios[tail_start - 1]
    ξ = _psis_smooth_tail!(tail, cutoff)

    # truncate at max of raw weights (1 after scaling)
    clamp!(is_ratios, 0, 1)
    # unsort the ratios to their original position:
    invpermute!(is_ratios, last.(ratio_index))

    return ξ::T
end


"""
    _def_tail_length(log_ratios::AbstractVector, r_eff::Real) -> tail_len::Integer

Define the tail length as in Vehtari et al. (2019).
"""
function _def_tail_length(length::Int, r_eff::Real)
    return Int(ceil(min(length / 5, 3 * sqrt(length / r_eff))))
end


"""
    _psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T) where {T<:Real} -> ξ::T

Takes an *already sorted* vector of observations from the tail and smooths it *in place*
with PSIS before returning shape parameter `ξ`.
"""
function _psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T) where {T <: Real}
    len = length(tail)
    if any(isinf.(tail))
        return ξ = Inf
    else
        @turbo @. tail = tail - cutoff

        # save time not sorting since tail is already sorted
        ξ, σ = gpdfit(tail)
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
function _generate_r_eff(
    weights::AbstractArray{R}, 
    dims::Base.AbstractVecOrTuple, 
    r_eff::T, 
    source::String,
)::T where {R<:Real, T<:AbstractVector{R}}
    output::T = similar(r_eff, dims[1])
    if isempty(r_eff)
        if source == "mcmc"
            @info "Adjusting for autocorrelation. If the posterior samples are not " *
                  "autocorrelated, specify the source of the posterior sample using the " *
                  "keyword argument `source`. MCMC samples are always autocorrelated; VI " *
                  "samples are not."
            output .= relative_eff(reshape(weights, dims))
        elseif source ∈ SAMPLE_SOURCES
            @info "Samples have not been adjusted for autocorrelation. If the posterior " *
                  "samples are autocorrelated, as in MCMC methods, ESS estimates will be " *
                  "upward-biased, and standard error estimates will be downward-biased. " *
                  "MCMC samples are always autocorrelated; VI samples are not."
            return output .= ones(R, dims[1])
        else
            throw(
                ArgumentError(
                    "$source is not a valid source. Valid sources are $SAMPLE_SOURCES."
                ),
            )
            return output .= ones(R, dims[1])
        end
    else 
        return r_eff
    end
    if any(_invalid_number, r_eff)
        throw(
            ArgumentError(
                "PSIS-LOO has encountered an error calculating ESS values for your " * 
                "Markov chains. Please check your inputs. $LIKELY_ERROR_CAUSES"
            )
        )
    end
    return output::T
end


"""
Make sure all inputs to `psis` are valid.
"""
function _check_input_validity_psis(
    log_ratios::AbstractArray{T, 3}, r_eff::AbstractVector{T}
) where {T <: Real}
    if any(_invalid_number, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains NaN  or inf values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `log_ratios` (array is empty)."))
    elseif length(r_eff) ≠ size(log_ratios, 1)
        throw(ArgumentError("Size of `r_eff` does not equal the number of data points."))
    end
    return nothing
end


"""
Check the tail to make sure a GPD fit is possible.
"""
function _check_tail(tail::AbstractVector{T}) where {T <: Real}
    if maximum(tail) ≈ minimum(tail)
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution: all tail values are the " *
                "same. Likely causes are: \n$LIKELY_ERROR_CAUSES",
            ),
        )
    elseif length(tail) < MIN_TAIL_LEN
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution: tail length was too " *
                "short. Likely causese are: \n$LIKELY_ERROR_CAUSES",
            ),
        )
    end
    return nothing
end


"""
Check if an input is invalid.
"""
function _invalid_number(x::Real)
    return isinf(x) || isnan(x)
end
