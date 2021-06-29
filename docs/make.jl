using ParetoSmooth
using Documenter

DocMeta.setdocmeta!(ParetoSmooth, :DocTestSetup, :(using ParetoSmooth); recursive=true)

makedocs(;
    modules=[ParetoSmooth],
    authors="Carlos Parada <paradac@carleton.edu>",
    repo="https://github.com/ParadaCarleton/ParetoSmooth.jl/blob/{commit}{path}#{line}",
    sitename="ParetoSmooth.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ParadaCarleton.github.io/ParetoSmooth.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ParadaCarleton/ParetoSmooth.jl",
)
