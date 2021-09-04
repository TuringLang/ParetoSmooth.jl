module ParetoSmooth
using Requires
using DocStringExtensions

function __init__()
    
    chains_loaded = false

    @require Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0" begin
        include("MCMCChainsHelpers.jl")
        include("TuringHelpers.jl")
        chains_loaded = true
    end 

    if !chains_loaded 
        @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" begin 
            include("MCMCChainsHelpers.jl")
        end
    end

end

include("AbstractCV.jl")
include("ESS.jl")
include("GPD.jl")
include("InternalHelpers.jl")
include("ImportanceSampling.jl")
include("LeaveOneOut.jl")
include("ModelComparison.jl")
include("NaiveLPD.jl")
include("PublicHelpers.jl")

end
