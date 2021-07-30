module ParetoSmooth
    using Requires

    function __init__()
        @require Turing = 
            "fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("TuringHelpers.jl")
        @require MCMCChains = 
            "c7f686f2-ff18-58e9-bc7b-31028e88f75d" include("MCMCChainsHelpers.jl")
    end

    include("ESS.jl")
    include("GPD.jl")
    include("InternalHelpers.jl")
    include("ImportanceSampling.jl")
    include("LooStructs.jl")
    include("LeaveOneOut.jl")
    include("Bootstrap.jl")
    include("PublicHelpers.jl")

end
