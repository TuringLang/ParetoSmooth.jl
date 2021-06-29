using PSIS
using Documenter

DocMeta.setdocmeta!(PSIS, :DocTestSetup, :(using PSIS); recursive=true)

makedocs(;
    modules=[PSIS],
    authors="Carlos Parada <paradac@carleton.edu>",
    repo="https://github.com/ParadaCarleton/PSIS.jl/blob/{commit}{path}#{line}",
    sitename="PSIS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ParadaCarleton.github.io/PSIS.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ParadaCarleton/PSIS.jl",
)
