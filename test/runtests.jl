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

@testset "ParetoSmooth.jl" begin

    # All of these should run
    
    with_rel_eff = psis(logLikelihoodArray, rel_eff)
    juliaPsis = psis(logLikelihoodArray)
    logLikelihoodMatrix = reshape(logLikelihoodArray, 32, 1000)
    chainIndex = vcat(fill(1, 500), fill(2, 500))
    matrixPsis = psis(logLikelihoodMatrix; chain_index=chainIndex)
    logPsis = psis(logLikelihoodArray; lw=true)


    # RMSE from R version is less than .1%
    @test sqrt(mean((with_rel_eff.weights ./ rWeights .- 1).^2)) ≤ .001
    # RMSE less than .2% when using InferenceDiagnostics' ESS
    @test sqrt(mean((juliaPsis.weights ./ rWeights .- 1).^2)) ≤ .002
    @test count(with_rel_eff.weights .≉ rWeights) ≤ 10
    @test count(juliaPsis.weights .≉ matrixPsis.weights) ≤ 10
    @test sqrt(mean((logPsis.weights .- log.(rWeights)).^2)) ≤ .001
end
