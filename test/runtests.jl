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
rPsis = RData.load("test/Psis_Object.RData")["psisObject"]
juliaPsis = psis(logLikelihoodArray)
juliaWeights = juliaPsis.weights

@testset "PSIS.jl" begin
    sqrt(mean((rWeights - juliaWeights).^2))  ≤ .001  # RMSE ≤ .001
end
