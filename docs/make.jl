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
        canonical = "https://jayren3996.github.io/GaussianFermions.jl",
        assets = ["assets/favicon.ico"],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
        "Manual" => [
            "Conventions" => "manual/conventions.md",
            "States" => "manual/states.md",
            "Hamiltonians & Time Evolution" => "manual/hamiltonians.md",
            "Lindblad Dynamics" => "manual/lindblad.md",
            "Quantum Trajectories" => "manual/trajectories.md",
            "Observables" => "manual/observables.md",
        ],
        "Examples" => [
            "Free-Fermion Chain" => "examples/free-fermion-chain.md",
            "Deterministic Lindblad" => "examples/deterministic-lindblad.md",
            "Trajectories vs Lindblad" => "examples/trajectories-vs-lindblad.md",
            "BdG Pairing" => "examples/bdg-pairing.md",
            "Monitored Mutual Information" => "examples/monitored-mutual-information.md",
        ],
        "API Reference" => [
            "Overview" => "reference/overview.md",
            "States & Modes" => "reference/states.md",
            "Hamiltonians" => "reference/hamiltonians.md",
            "Lindblad & Channels" => "reference/dynamics.md",
            "Trajectory Primitives" => "reference/trajectories.md",
            "Observables" => "reference/observables.md",
        ],
    ],
)
