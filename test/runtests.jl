using PSIS
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
juliaPsis = psis(logLikelihoodArray)
relEffSpecified = psis(logLikelihoodArray, rel_eff)

@testset "PSIS.jl" begin
    @test sqrt(mean((relEffSpecified.weights - rWeights).^2)) ≤ .0001  # RMSE ≤ .0001 (~10%)
end
