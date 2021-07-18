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