using ParetoSmooth
using Test


@testset "ParetoSmooth.jl" begin
    
    # Tests must be kept in this order
    include("tests/BasicTests.jl")
    include("tests/TuringTests.jl")
    include("tests/ComparisonTests.jl")

end
