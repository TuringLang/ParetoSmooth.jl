using MCMCDiagnosticTools
using Tullio

export relative_eff, psis_ess, sup_ess

"""
    relative_eff(sample::AbstractArray{<:Real, 3}; [method])

Calculate the relative efficiency of an MCMC chain, i.e. the effective sample size divided
by the nominal sample size. If none is provided, use the default method from 
MCMCDiagnosticTools.
"""
function relative_eff(sample::AbstractArray{<:Real, 3}; maxlag=size(sample, 2), kwargs...)
    dims = size(sample)
    post_sample_size = dims[2] * dims[3]
    ess_sample = inv.(permutedims(sample, [2, 1, 3]))
    ess, = MCMCDiagnosticTools.ess_rhat(ess_sample; maxlag=maxlag, kwargs...)
    r_eff = ess / post_sample_size
    return r_eff
end


"""
    function psis_ess(
        weights::AbstractVector{<:Real},
        r_eff::AbstractVector{<:Real}
    ) -> AbstractVector{<:Real}

Calculate the (approximate) effective sample size of a PSIS sample, using the correction in
Vehtari et al. 2019. This uses the variance-based definition of ESS, and measures the L2 
distance of the proposal and target distributions.

# Arguments

  - `weights`: A set of importance sampling weights derived from PSIS.
  - `r_eff`: The relative efficiency of the MCMC chains from which PSIS samples were derived.

See `?relative_eff` to calculate `r_eff`.
"""
function psis_ess(
    weights::AbstractMatrix{T}, r_eff::AbstractVector{T}
) where {T <: Union{Real, Missing}}
    @tullio sum_of_squares[x] := weights[x, y]^2
    return r_eff ./ sum_of_squares
end


function psis_ess(weights::AbstractMatrix{<:Real})
    @warn "PSIS ESS not adjusted based on MCMC ESS. MCSE and ESS estimates " *
          "will be overoptimistic if samples are autocorrelated."
    return psis_ess(weights, ones(size(weights)))
end


"""
    function sup_ess(
        weights::AbstractVector{<:Real},
        r_eff::AbstractVector{<:Real}
    ) -> AbstractVector

Calculate the supremum-based effective sample size of a PSIS sample, i.e. the inverse of the
maximum weight. This measure is more trustworthy than the `ess` from `psis_ess`. It uses the
L-∞ norm.

# Arguments
  - `weights`: A set of importance sampling weights derived from PSIS.
  - `r_eff`: The relative efficiency of the MCMC chains from which PSIS samples were 
    derived.
"""
function sup_ess(
    weights::AbstractMatrix{T}, r_eff::V
) where {T<:Real, V<:AbstractVector{T}}
    return inv.(dropdims(maximum(weights; dims=2); dims=2)) .* r_eff
end
