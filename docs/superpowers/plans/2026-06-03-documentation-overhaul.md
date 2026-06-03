# Documentation Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the thin README-style docs with a conventions-first Documenter.jl site for physics researchers, backed by grouped API reference pages and public docstrings.

**Architecture:** Keep Documenter.jl and the existing package API. Reorganize docs into orientation, manual, examples, and API reference; add focused docstrings in source files so `@docs` pages are useful. Verification is a package test run plus `julia --project=docs docs/make.jl`.

**Tech Stack:** Julia, Documenter.jl, Markdown, GaussianFermions.jl source docstrings.

---

### Task 1: Documenter Navigation And Site Shape

**Files:**
- Modify: `docs/make.jl`
- Modify: `docs/src/index.md`
- Create: `docs/src/manual/conventions.md`
- Create: `docs/src/manual/lindblad.md`
- Create: `docs/src/examples/free-fermion-chain.md`
- Create: `docs/src/examples/deterministic-lindblad.md`
- Create: `docs/src/examples/trajectories-vs-lindblad.md`
- Create: `docs/src/examples/bdg-pairing.md`
- Create: `docs/src/examples/monitored-mutual-information.md`

- [ ] **Step 1: Update `docs/make.jl` page tree**

Use this page layout:

```julia
pages = [
    "Home" => "index.md",
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
]
```

- [ ] **Step 2: Rewrite `docs/src/index.md`**

The home page must explain what the package simulates, include a representation/dynamics decision table, and link to the manual, examples, and API reference.

- [ ] **Step 3: Add `docs/src/manual/conventions.md`**

The page must define site indexing, `C`, `F`, Majorana basis order, covariance sign convention, Lindblad amplitude-vector convention, dephasing mode/rate convention, and bang-function mutation convention.

- [ ] **Step 4: Add examples directory pages**

Create the five example files listed above. Keep examples short and deterministic where `@example` is used; longer stochastic workflows should be plain `julia` code blocks with fixed RNG seeds.

- [ ] **Step 5: Run a fast docs build smoke test**

Run:

```bash
julia --color=yes --project=docs docs/make.jl
```

Expected: build exits 0. Cross-reference errors from renamed pages must be fixed before Task 2.

### Task 2: Manual Rewrite

**Files:**
- Modify: `docs/src/manual/states.md`
- Modify: `docs/src/manual/hamiltonians.md`
- Modify: `docs/src/manual/lindblad.md`
- Modify: `docs/src/manual/trajectories.md`
- Modify: `docs/src/manual/observables.md`
- Leave unlinked legacy page: `docs/src/manual/correlation-lindblad.md`
- Leave unlinked legacy page: `docs/src/manual/majorana-bdg.md`

- [ ] **Step 1: Rewrite `manual/states.md`**

Cover `SlaterState`, `CorrelationState`, `MajoranaState`, conversion rules, purity, and `nmodes`. Start from representation choice and conventions; do not start from constructor lists.

- [ ] **Step 2: Rewrite `manual/hamiltonians.md`**

Cover `QuadraticHamiltonian`, `hopping`, `chemical_potential`, `propagator`, `BdGHamiltonian`, `groundstate`, `thermalstate`, and `quasiparticle_energies`. Make clear when number-conserving Hamiltonians lift into BdG form.

- [ ] **Step 3: Write `manual/lindblad.md`**

Merge the useful content from current `correlation-lindblad.md` and `majorana-bdg.md`. Explain the `C` and `Γ` closure levels, loss/gain amplitude vectors, dephasing pairs, pairing baths, `steadystate`, and dense finite-system scope.

- [ ] **Step 4: Rewrite `manual/trajectories.md`**

Preserve the core message that there is no trajectory runner. Explain projective, MCWF, and QSD primitives for number-conserving and Majorana states, with short loops and links to examples.

- [ ] **Step 5: Rewrite `manual/observables.md`**

Explain observables by what matrix they come from and how they are used in monitored dynamics. Include density, correlations, number diagnostics, entropies, spectra, mutual information, and tripartite information.

- [ ] **Step 6: Run docs build**

Run:

```bash
julia --color=yes --project=docs docs/make.jl
```

Expected: build exits 0. Any broken page links or malformed math must be fixed before Task 3.

### Task 3: Grouped API Reference Pages

**Files:**
- Modify: `docs/src/reference/overview.md`
- Modify: `docs/src/reference/states.md`
- Modify: `docs/src/reference/hamiltonians.md`
- Modify: `docs/src/reference/dynamics.md`
- Create: `docs/src/reference/trajectories.md`
- Modify: `docs/src/reference/observables.md`

- [ ] **Step 1: Replace `@autodocs` with grouped `@docs` blocks**

Use topic groups instead of source-order dumps. Reference pages should include only exported public API.

- [ ] **Step 2: Use these groups**

States:

````markdown
## State Types
```@docs
SlaterState
CorrelationState
MajoranaState
```

## Modes And Accessors
```@docs
QuasiMode
nmodes
ispure
correlation_matrix
correlation
covariance_matrix
fermion_correlations
normal_correlation
anomalous_correlation
thermalstate
maximally_mixed
```
````

Hamiltonians:

````markdown
```@docs
QuadraticHamiltonian
BdGHamiltonian
hopping
chemical_potential
propagator
groundstate
quasiparticle_energies
```
````

Lindblad & channels:

````markdown
```@docs
CorrelationLindblad
MajoranaLindblad
lindblad_rhs
majorana_lindblad_rhs
steadystate
OccupationMonitor
HoleMonitor
Loss
Gain
dephasing
loss
gain
```
````

Trajectory primitives:

````markdown
```@docs
evolve
evolve!
Gate
apply!
measure!
weak_measure!
Feedback
jump_rate
apply_click!
noclick_operator
apply_noclick!
NonHermitianGenerator
effective_hamiltonian
noclick_propagator
evolve_noclick!
loss_jump
gain_jump
majorana_jump
```
````

Observables:

````markdown
```@docs
density
density_profile
particle_number
number_variance
purity
parity
correlation_profile
entanglement_entropy
renyi_entropy
mutual_information
tripartite_information
entanglement_spectrum
entanglement_hamiltonian_spectrum
```
````

- [ ] **Step 3: Run docs build**

Run:

```bash
julia --color=yes --project=docs docs/make.jl
```

Expected: build exits 0. Missing-doc warnings should guide Task 4.

### Task 4: Public Docstring Pass

**Files:**
- Modify: `src/States.jl`
- Modify: `src/Modes.jl`
- Modify: `src/Hamiltonians.jl`
- Modify: `src/BdGHamiltonians.jl`
- Modify: `src/MajoranaStates.jl`
- Modify: `src/Observables.jl`
- Modify: `src/MajoranaObservables.jl`
- Modify: `src/CorrelationLindblad.jl`
- Modify: `src/MajoranaLindblad.jl`
- Modify: `src/Channels.jl`
- Modify: `src/Trajectory.jl`
- Modify: `src/MajoranaTrajectory.jl`

- [ ] **Step 1: Add state and mode docstrings**

Add concise docstrings before public constructors and accessor functions in `States.jl`, `Modes.jl`, and `MajoranaStates.jl`. Each docstring must state the relevant convention and whether returned matrices are copies.

- [ ] **Step 2: Add Hamiltonian docstrings**

Add docstrings for `QuadraticHamiltonian`, `hopping`, `chemical_potential`, `propagator`, `BdGHamiltonian`, `groundstate`, and `quasiparticle_energies`. Include the Hamiltonian equation and the evolution convention.

- [ ] **Step 3: Add Lindblad/channel docstrings**

Add docstrings for `CorrelationLindblad`, `MajoranaLindblad`, RHS functions, `steadystate`, channel types, and channel constructors. State rate and amplitude-vector conventions.

- [ ] **Step 4: Add trajectory docstrings**

Add docstrings for evolution, measurement, weak measurement, click/no-click primitives, non-Hermitian helpers, feedback, and Majorana jump constructors. State mutation behavior and whether a value is a rate or probability.

- [ ] **Step 5: Add observable docstrings**

Add docstrings for density, particle-number diagnostics, correlations, entropies, spectra, and information measures. State region argument conventions and whether values are computed from `C` or `Γ`.

- [ ] **Step 6: Run docs build**

Run:

```bash
julia --color=yes --project=docs docs/make.jl
```

Expected: build exits 0 and grouped `@docs` pages render without missing binding errors.

### Task 5: Verification And Cleanup

**Files:**
- Inspect: all changed docs and source files
- Inspect: `docs/build` status after build
- Inspect: `docs/Manifest.toml` status after instantiation

- [ ] **Step 1: Run package tests**

Run:

```bash
julia --color=yes --project=. -e 'using Pkg; Pkg.test()'
```

Expected: test suite exits 0.

- [ ] **Step 2: Run final docs build**

Run:

```bash
julia --color=yes --project=docs docs/make.jl
```

Expected: docs build exits 0.

- [ ] **Step 3: Check git status**

Run:

```bash
git status --short
```

Expected: source docs/source files changed; ignored/generated files such as `docs/Manifest.toml` and `docs/build` are not staged.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add docs/src docs/make.jl src
git commit -m "docs: overhaul package documentation"
```

Expected: commit succeeds on branch `codex/docs-overhaul`.
