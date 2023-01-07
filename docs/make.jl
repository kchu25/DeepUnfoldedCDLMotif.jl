using DeepUnfoldedCDLMotif
using Documenter

DocMeta.setdocmeta!(DeepUnfoldedCDLMotif, :DocTestSetup, :(using DeepUnfoldedCDLMotif); recursive=true)

makedocs(;
    modules=[DeepUnfoldedCDLMotif],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    repo="https://github.com/kchu25/DeepUnfoldedCDLMotif.jl/blob/{commit}{path}#{line}",
    sitename="DeepUnfoldedCDLMotif.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kchu25.github.io/DeepUnfoldedCDLMotif.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/DeepUnfoldedCDLMotif.jl",
    devbranch="main",
)
