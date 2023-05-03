module ParetoSmooth

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
    @require MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d" begin
        include("../ext/ParetoSmoothMCMCChainsExt.jl")
    end
    @require DynamicPPL = "366bfd00-2699-11ea-058f-f148b4cae6d8" begin
        include("../ext/ParetoSmoothDynamicPPLExt.jl")
    end
end
end

@static if VERSION >= v"1.8"
@inline exp_inline(x) = @inline exp(x)
else
const exp_inline = exp
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
