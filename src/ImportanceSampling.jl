using Tullio

const LIKELY_ERROR_CAUSES = """
1. Incorrect inputs -- check your program for bugs. If you provided an `r_eff` argument,
double check it is correct.
2. Your chains failed to converge. Check diagnostics. 
3. You do not have enough posterior samples (ESS < ~25).
"""
const MIN_TAIL_LEN = 5  # Minimum size of a tail for PSIS to give sensible answers
const SAMPLE_SOURCES = ["mcmc", "vi", "other"]

export psis, psis!, Psis


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
    R <: Real,
    AT <: AbstractArray{R, 3},
    VT <: AbstractVector{R}
}
    weights::AT
    pareto_k::VT
    ess::VT
    sup_ess::VT
    r_eff::VT
    tail_len::AbstractVector{Int}
    posterior_sample_size::Int
    data_size::Int
end


function Base.getproperty(psis_obj::Psis, k::Symbol)
    if k === :log_weights
        return log.(getfield(psis_obj, :weights))
    else
        return getfield(psis_obj, k)
    end
end


function Base.propertynames(psis_object::Psis)
    return (
        fieldnames(typeof(psis_object))...,
        :log_weights,
    )
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
        r_eff::AbstractVector{T}; 
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
  - `calc_ess::Bool=true`: If `false`, do not calculate ESS diagnostics. Attempting to
    access ESS diagnostics will return an empty array.
  - `checks::Bool=true`: If `true`, check inputs for possible errors. Disabling will improve 
    performance slightly.

See also: [`relative_eff`]@ref, [`psis_loo`]@ref, [`psis_ess`]@ref.
"""
function psis(
    log_ratios::AbstractArray{T, 3};
    source::Union{AbstractString, Symbol}="default",
    r_eff::AbstractVector{T}=relative_eff(log_ratios; source=source),
    calc_ess::Bool = true, 
    skip_checks::Bool = false
) where T <: Real

    dims = size(log_ratios)
    data_size = dims[1]
    post_sample_size = dims[2] * dims[3]

    skip_checks || _check_input_validity_psis(log_ratios)

    source = lowercase(String(source))
    if source == "default"
        @info "No source provided for samples; variables are assumed to be from a Markov " *
        "Chain. If the samples are independent, specify this with keyword argument " *
        "`source=:other`."
    end

    if !skip_checks && size(r_eff, 1) ≠ data_size
        throw(ArgumentError("Size of `r_eff` does not equal the number of data points."))
    end

    # Reshape to matrix (easier to deal with)
    
    weights = similar(log_ratios)
    weights_mat = reshape(weights, data_size, post_sample_size)
    @. weights = exp(log_ratios - $maximum(log_ratios; dims=(2,3)))


    tail_length = similar(r_eff, Int)
    ξ = similar(r_eff)
    @inbounds @views Threads.@threads for i in eachindex(tail_length)
        tail_length[i] = _def_tail_length(post_sample_size, r_eff[i])
        ξ[i] = psis!(
            weights_mat[i, :], r_eff[i]; 
            tail_length=tail_length[i], log_weights = false
        )
    end

    @tullio norm_const[i] := weights[i, j, k]
    @. weights = weights / norm_const

    
    if calc_ess
        ess = psis_ess(weights_mat, r_eff)
        inf_ess = sup_ess(weights_mat, r_eff)
    else
        ess = similar(weights_mat, 0)
        inf_ess = similar(weights_mat, 0)
    end

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
    chain_index::AbstractVector=_assume_one_chain(log_ratios),
    kwargs...,
)
    chain_index = Vector(Int.(chain_index))
    new_log_ratios = _convert_to_array(log_ratios, chain_index)
    return psis(new_log_ratios; kwargs...)
end


function psis(is_ratios::AbstractVector{<:Real}, args...; kwargs...)
    new_ratios = copy(is_ratios)
    ξ = psis!(new_ratios, kwargs...)
    return new_ratios, ξ
end



"""
    psis!(is_ratios::AbstractVector{<:Real}; tail_length::Integer, log_ratios=false) -> Real

Do PSIS on a single vector, smoothing its tail values *in place* before returning the 
estimated shape constant for the `pareto_k` distribution. This *does not* normalize the 
log-weights.

# Arguments

  - `is_ratios::AbstractVector{<:Real}`: A vector of importance sampling ratios,
    scaled to have a maximum of 1.
  - `r_eff::AbstractVector{<:Real}`: The relative effective sample size, used to calculate
    the effective sample size. See [rel_eff]@ref for more information.
  - `log_weights::Bool`: A boolean indicating whether the input vector is a vector of log
    ratios, rather than raw importance sampling ratios.

# Returns

  - `Real`: ξ, the shape parameter for the GPD. Bigger numbers indicate thicker tails.

# Notes

Unlike the methods for arrays, `psis!` performs no checks to make sure the input values are 
valid.
"""
function psis!(is_ratios::AbstractVector{T}, r_eff::T=one(T);
    log_weights::Bool = true,
    tail_length::Integer = _def_tail_length(length(is_ratios), r_eff),
    skip_checks::Bool = false
) where T<:Real

    skip_checks || _check_input_validity_psis(is_ratios)
    
    len = length(is_ratios)
    tail_start = len - tail_length + 1  # index of smallest tail value

    # sort is_ratios and also get results of sortperm() at the same time
    ratio_index = collect(zip(is_ratios, Base.OneTo(len)))
    partialsort!(ratio_index, (tail_start-1):len; by=first)
    is_ratios .= first.(ratio_index)
    @views tail = is_ratios[tail_start:len]
    _check_tail(tail)
    if log_weights 
        biggest = maximum(tail)
        @. tail = exp(tail - biggest)
    end

    # Get value just before the tail starts:
    cutoff = is_ratios[tail_start - 1]
    ξ = _psis_smooth_tail!(tail, cutoff, r_eff)

    # truncate at max of raw weights (1 after scaling)
    clamp!(is_ratios, 0, 1)
    # unsort the ratios to their original position:
    invpermute!(is_ratios, last.(ratio_index))

    if log_weights 
        @. tail = log(tail + biggest)
    end

    return ξ
end


"""
    _def_tail_length(log_ratios::AbstractVector, r_eff::Real) -> Integer

Define the tail length as in Vehtari et al. (2019), with the small addition that the tail
must a multiple of `32*bit_length` (which improves performance).
"""
function _def_tail_length(length::Integer, r_eff::Real=one(T))
    return min(cld(length, 5), ceil(3 * sqrt(length / r_eff))) |> Int
end


"""
    _psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T, r_eff::T=1) where {T<:Real} 
    -> ξ::T

Takes an *already sorted* vector of observations from the tail and smooths it *in place*
with PSIS before returning shape parameter `ξ`.
"""
function _psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T, r_eff::T=one(T)) where {T <: Real}
    len = length(tail)
    if any(isinf.(tail))
        return ξ = Inf
    else
        @. tail = tail - cutoff

        # save time not sorting since tail is already sorted
        ξ, σ = gpd_fit(tail, r_eff)
        @. tail = gpd_quantile(($(1:len) - 0.5) / len, ξ, σ) + cutoff
    end
    return ξ
end



##########################
#### HELPER FUNCTIONS ####
##########################


"""
Make sure all inputs to `psis` are valid.else
"""
function _check_input_validity_psis(
    log_ratios::AbstractArray{<:Real}
)
    if any(_invalid_number, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains NaN  or inf values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `log_ratios` (array is empty)."))
    end
    return nothing
end


"""
Check the tail to make sure a GPD fit is possible.
"""
function _check_tail(tail::AbstractVector{T}) where {T <: Real}
    if tail[end] ≈ tail[1]
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution; all tail values are the " *
                "same. Likely causes are:\n$LIKELY_ERROR_CAUSES",
            ),
        )
    elseif length(tail) ≤ MIN_TAIL_LEN
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution; tail length was too " *
                "short. Likely causes are:\n$LIKELY_ERROR_CAUSES",
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
