using ParetoSmooth
using Documenter

makedocs(;
    modules=[ParetoSmooth],
    authors="Carlos Parada <paradac@carleton.edu>",
    repo="https://github.com/TuringLang/ParetoSmooth.jl/blob/{commit}{path}#{line}",
    sitename="ParetoSmooth.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://turinglang.github.io/ParetoSmooth.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    strict=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/TuringLang/ParetoSmooth.jl",
    push_preview=true,
    devbranch="main",
)
