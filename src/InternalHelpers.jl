const CHAIN_INDEX_DOC = """
`chain_index::Vector`: An optional vector of integers specifying which chain each 
    step belongs to. For instance, `chain_index[step]` should return `2` if
    `log_likelihood[:, step]` belongs to the second chain.
"""

const LIKELIHOOD_FUNCTION_ARG = """
`ll_fun::Function`: A function taking a single data point and returning the log-likelihood 
of that point. This function must take the form `f(θ[1], ..., θ[n], data)`, where `θ` is the
parameter vector. See also the `splat` keyword argument.
"""

const R_EFF_DOC = """
`r_eff::AbstractArray{T}`: An (optional) vector of relative effective sample sizes used 
in ESS calculations. If left empty, calculated automatically using the FFTESS method 
from InferenceDiagnostics.jl. See `relative_eff` to calculate these values.
"""

const ARGS = """
`args...`: Positional arguments to be passed to
"""

const KWARGS = """
`kwargs...`: Keyword arguments to be passed to
"""


###############
## FUNCTIONS ##
###############

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
Throw a warning if any `pareto_k` values exceed 0.7.
"""
function _throw_pareto_k_warning(ξ)
    if any(ξ .≥ 1)
        @warn "Some Pareto k values are extremely high (>1). PSIS will not produce " *
              "consistent estimates."
    elseif any(ξ .≥ 0.7)
        @warn "Some Pareto k values are high (>.7), indicating that PSIS has failed to " *
              "approximate the true distribution."
    elseif any(ξ .≥ 0.5)
        @info "Some Pareto k values are slightly high (>.5); some pointwise estimates " *
              "may be slow to converge or have high variance."
    end
end


"""
Assume that all objects belong to a single chain if chain index is missing. Inform user.
"""
function _assume_one_chain(matrix)
    @info "Chain information was not provided; " *
          "all samples are assumed to be drawn from a single chain."
    return ones(length(matrix))
end


"""
Convert a matrix+chain_index representation to a 3d array representation.
"""
function _convert_to_array(matrix::AbstractMatrix, chain_index::AbstractVector)
    indices = unique(chain_index)
    biggest_idx = maximum(indices)
    dims = size(matrix)
    if dims[2] ≠ length(chain_index)
        throw(ArgumentError("Some entries do not have a chain index."))
    elseif !issetequal(indices, 1:biggest_idx)
        throw(
            ArgumentError(
                "Indices must be numbered from 1 through the total number of chains."
            ),
        )
    else
        # Check how many elements are in each chain, assign to "counts"
        counts = count.(eachslice(chain_index .== indices'; dims=2))
        # check if all inputs are the same length
        if !all(==(counts[1]), counts)
            throw(ArgumentError("All chains must be of equal length."))
        end
    end
    new_ratios = similar(matrix, dims[1], dims[2] ÷ biggest_idx, biggest_idx)
    for i in 1:biggest_idx
        new_ratios[:, :, i] .= matrix[:, chain_index .== i]
    end
    return new_ratios
end