using Documenter, TotalUnimodularity

DocMeta.setdocmeta!(TotalUnimodularity, :DocTestSetup, :(using TotalUnimodularity);
                    recursive = true)

makedocs(;
    modules = [TotalUnimodularity],
    sitename = "TotalUnimodularity.jl",
    authors = "Michael McVeagh",
    format = Documenter.HTML(; canonical = "https://mcvmcv.github.io/TotalUnimodularity.jl",
                             edit_link = "main"),
    pages = ["Home" => "index.md", "API" => "api.md"],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/mcvmcv/TotalUnimodularity.jl", devbranch = "main")
