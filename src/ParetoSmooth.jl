module ParetoSmooth
    using Requires

    function __init__()
        @require Turing="fce5fe82-541a-59a6-adf8-730c64b5f9a0" include("turing_utilities.jl")
    end

    include("ESS.jl")
    include("GPD.jl")
    include("ImportanceSampling.jl")
    include("LooStructs.jl")
    include("LeaveOneOut.jl")
    include("utilities.jl")

end
