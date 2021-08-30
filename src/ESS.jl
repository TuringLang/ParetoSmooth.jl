using FFTW
using LoopVectorization
using MCMCDiagnosticTools

using Tullio

export relative_eff, psis_ess, psis_n_eff

"""
    relative_eff(
        sample::AbstractArray{Real, 3}; 
        method=MCMCDiagnosticTools.FFTESSMethod()
    )

Calculate the relative efficiency of an MCMC chain, i.e. the effective sample size divided
by the nominal sample size.
"""
function relative_eff(
    sample::AbstractArray{T, 3}; method=MCMCDiagnosticTools.FFTESSMethod()
) where {T <: Union{Real, Missing}}
    dims = size(sample)
    post_sample_size = dims[2] * dims[3]
    ess_sample = inv.(permutedims(sample, [2, 1, 3]))
    ess, = MCMCDiagnosticTools.ess_rhat(ess_sample; method=method, maxlag=dims[2])
    r_eff = ess / post_sample_size
    return r_eff
end


"""
    function psis_ess(
        weights::AbstractVector{T},
        r_eff::AbstractVector{T}
    ) -> AbstractVector{T}

Calculate the (approximate) effective sample size of a PSIS sample, using the correction in
Vehtari et al. 2019.

# Arguments

  - `weights`: A set of importance sampling weights derived from PSIS.
  - `r_eff`: The relative efficiency of the MCMC chains from which PSIS samples were derived.

See `?relative_eff` to calculate `r_eff`.
"""
function psis_ess(
    weights::AbstractVector{T}, r_eff::AbstractVector{T}
) where {T <: Union{Real, Missing}}
    @tullio sum_of_squares := weights[x]^2
    return r_eff ./ sum_of_squares
end


function psis_ess(
    weights::AbstractMatrix{T}, r_eff::AbstractVector{T}
) where {T <: Union{Real, Missing}}
    @tullio sum_of_squares[x] := weights[x, y]^2
    return @tturbo r_eff ./ sum_of_squares
end


function psis_ess(weights::AbstractMatrix{T}) where {T <: Union{Real, Missing}}
    @warn "PSIS ESS not adjusted based on MCMC ESS. MCSE and ESS estimates " *
          "will be overoptimistic if samples are autocorrelated."
    return psis_ess(weights, ones(size(weights)))
end


function psis_n_eff(args...; kwargs...)  # Alias for compatibility with R version
    return psis_ess(args...; kwargs...)
end