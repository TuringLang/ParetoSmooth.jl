using ParetoSmooth
using Test


@testset "ParetoSmooth.jl" begin
    
    include("tests/BasicTests.jl")
    include("tests/TuringTests.jl")
    include("tests/ComparisonTests.jl")

end
