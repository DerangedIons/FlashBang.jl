using FlashBang
using Documenter
using DocumenterVitepress

DocMeta.setdocmeta!(FlashBang, :DocTestSetup, :(using FlashBang); recursive=true)

makedocs(;
    modules=[FlashBang],
    authors="Kyle Beggs (beggskw@gmail.com) and contributors",
    sitename="FlashBang.jl",
    repo=Documenter.Remotes.GitHub("DerangedIons", "FlashBang.jl"),
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="https://github.com/DerangedIons/FlashBang.jl",
        devbranch="main",
        devurl="dev",
        build_vitepress=(!haskey(ENV, "VITEPRESS_DEV")),
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Guides" => [
            "The Solution Vector" => "guides/solution_vector.md",
            "Saving Output" => "guides/saving.md",
            "Running on a GPU" => "guides/gpu.md",
            "Thunderbolt Compatibility" => "guides/thunderbolt.md",
        ],
        "Examples" => "examples.md",
        "Reference" => [
            "Things That Will Bite You" => "reference/pitfalls.md",
            "API" => "reference/api.md",
        ],
    ],
    clean=false,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/DerangedIons/FlashBang.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
)
