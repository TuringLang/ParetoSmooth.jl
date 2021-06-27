using JuLOOa
using Documenter

DocMeta.setdocmeta!(JuLOOa, :DocTestSetup, :(using JuLOOa); recursive=true)

makedocs(;
    modules=[JuLOOa],
    authors="Carlos Parada <paradac@carleton.edu>",
    repo="https://github.com/ParadaCarleton/JuLOOa.jl/blob/{commit}{path}#{line}",
    sitename="JuLOOa.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ParadaCarleton.github.io/JuLOOa.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ParadaCarleton/JuLOOa.jl",
)
