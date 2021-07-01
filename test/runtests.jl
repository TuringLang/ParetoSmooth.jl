using ParetoSmooth
using Test
using Statistics

import RData

let ogArray = RData.load("test/Example_Log_Likelihood_Array.RData")["x"]
    global logLikelihoodArray = copy(permutedims(ogArray, [3, 1, 2]))
end
let ogWeights = RData.load("test/weightMatrix.RData")["weightMatrix"]
    global rWeights = exp.(permutedims(reshape(ogWeights, 500, 2, 32), [3, 1, 2]))
end
rel_eff = RData.load("test/Rel_Eff.RData")["rel_eff"]
rPsis = RData.load("test/Psis_Object.RData")["psisObject"]
relEffSpecified = psis(logLikelihoodArray, rel_eff)
juliaPsis = psis(logLikelihoodArray)
logLikelihoodMatrix = reshape(logLikelihoodArray, 32, 1000)
chainIndex = vcat(fill(1, 500), fill(2, 500))
matrixPsis = psis(logLikelihoodMatrix; chain_index=chainIndex)
logPsis = psis(logLikelihoodArray; lw=true)

@testset "ParetoSmooth.jl" begin
    # Difference from R version is less than .1%
    @test mean((relEffSpecified.weights ./ rWeights .- 1).^2) ≤ .001
    # Difference less than .2% when using InferenceDiagnostics' ESS
    @test mean((juliaPsis.weights ./ rWeights .- 1).^2) ≤ .002 
    @test juliaPsis.weights == matrixPsis.weights
    @test mean((logPsis.weights .- log.(rWeights)).^2) ≤ .0001 
end
