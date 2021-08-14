using ParetoSmooth
using Test

using AxisKeys
using NamedDims
using Statistics

import RData


@testset "ParetoSmooth.jl" begin
    
    # Tests must be kept in this order
    include("BasicTests.jl")
    include("TuringTests.jl")
    include("ComparisonTests.jl")

end
