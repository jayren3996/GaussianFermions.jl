using Documenter
using GaussianFermions

DocMeta.setdocmeta!(
    GaussianFermions,
    :DocTestSetup,
    :(using GaussianFermions),
    recursive = true,
)

makedocs(
    sitename = "GaussianFermions.jl",
    authors = "JieRen and contributors",
    modules = [GaussianFermions],
    checkdocs = :none,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://jayren3996.github.io/GaussianFermions.jl/stable",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Manual" => [
            "States" => "manual/states.md",
            "Hamiltonians & Time Evolution" => "manual/hamiltonians.md",
            "Observables" => "manual/observables.md",
            "Correlation-Matrix Lindblad" => "manual/correlation-lindblad.md",
            "Majorana / BdG Foundation" => "manual/majorana-bdg.md",
            "Quantum Trajectories" => "manual/trajectories.md",
        ],
        "API Reference" => [
            "Overview" => "reference/overview.md",
            "States & Spectra" => "reference/states.md",
            "Hamiltonians" => "reference/hamiltonians.md",
            "Observables" => "reference/observables.md",
            "Dynamics" => "reference/dynamics.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/jayren3996/GaussianFermions.jl",
    devbranch = "main",
    push_preview = true,
)
