module ESS

using InferenceDiagnostics, LoopVectorization

export relative_eff

"""
    relative_eff(sample::AbstractArray, cores = Threads.nthreads())
Compute the MCMC effective sample size divided by the actual sample size.
"""
function relative_eff(sample::AbstractArray{T <: Real, 3}, cores::Integer = Threads.nthreads(), ...)
    dimensions = size(sample)
    chainCount = dimensions[2]
    sampleSize = dimensions[2] * dimensions[1]
    rEff = zeroes(chainCount)
    @tturbo for i in 1:chainCount
      rEff[i] = ess_rhat(sample)[1] ./ sampleSize
    end
    return rEff
end

function relative_eff(sample::AbstractArray{T <: Real, 2}, cores::Integer = Threads.nthreads(), ...)
    dimensions = size(sample)
    return relative_eff(reshape(sample, dimensions[1], 1, dimensions[2]), cores, ...)
end


function psis_n_eff(weights::AbstractVector{T <: Real}, r_eff::Real = 1)
    if r_eff == 1
      @warn "PSIS n_eff not adjusted based on MCMC n_eff. PSIS MCSE estimates will be overoptimistic."
    end
    
    return @tullio grad=false effectiveSample := weights[x] |> r_eff / _
end

end