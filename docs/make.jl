using FlashBang
using Documenter

DocMeta.setdocmeta!(FlashBang, :DocTestSetup, :(using FlashBang); recursive=true)

makedocs(;
    modules=[FlashBang],
    authors="Kyle Beggs (beggskw@gmail.com) and contributors",
    sitename="FlashBang.jl",
    format=Documenter.HTML(;
        canonical="https://DerangedIons.github.io/FlashBang.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DerangedIons/FlashBang.jl",
    devbranch="main",
)
