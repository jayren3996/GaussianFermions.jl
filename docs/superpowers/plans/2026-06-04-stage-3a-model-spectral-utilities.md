# Stage 3A Model And Spectral Utilities Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight, dependency-free model-construction and spectral-analysis layer for common finite free-fermion and BdG workflows.

**Architecture:** Keep the package identity as a finite Gaussian-state engine. Add standard finite model constructors in one focused source file and add small spectral helpers that operate on the existing `QuadraticHamiltonian` and `BdGHamiltonian` types. Do not introduce a symbolic lattice DSL, external topology dependency, trajectory ensemble runner, or sparse backend in this slice.

**Tech Stack:** Julia, LinearAlgebra dense eigensolvers, existing `QuadraticHamiltonian` / `BdGHamiltonian` APIs, existing `Test` suite, Documenter docs.

---

## File Structure

- Create `src/Models.jl`: finite constructors `ssh_chain`, `aubry_andre_chain`, `kitaev_chain`, plus `energy_spectrum` and `bloch_bands`.
- Modify `src/GaussianFermions.jl`: include `Models.jl` after `BdGHamiltonians.jl`, because the constructors return existing Hamiltonian types.
- Modify `test/runtests.jl`: add focused tests for model matrices, BdG conversion, spectra, and validation errors.
- Modify `docs/src/reference/hamiltonians.md`: add the new model and spectral helper docstrings.
- Modify `docs/src/manual/hamiltonians.md`: add a short section that distinguishes finite model constructors from a full lattice/model DSL.
- Modify `docs/src/examples/kitaev-chain.md`: replace its local `kitaev_blocks` helper with the exported `kitaev_chain` constructor.

Do not modify trajectory APIs in this slice. The current source, docs, and tests explicitly say trajectory averaging is caller-owned.

---

### Task 1: Finite Model Constructors

**Files:**
- Create: `src/Models.jl`
- Modify: `src/GaussianFermions.jl`
- Test: `test/runtests.jl`

- [ ] **Step 1: Write failing constructor tests**

Add this testset after `"Hamiltonian & evolution"` in `test/runtests.jl`:

```julia
    @testset "Model constructors" begin
        Hssh = ssh_chain(4; t1=1.0, t2=0.25, pbc=false)
        @test Matrix(Hssh) ≈ ComplexF64[
            0    1.0  0     0
            1.0  0    0.25  0
            0    0.25 0     1.0
            0    0    1.0   0
        ]

        Hssh_pbc = ssh_chain(4; t1=1.0, t2=0.25, pbc=true)
        @test Matrix(Hssh_pbc)[4, 1] ≈ 0.25
        @test Matrix(Hssh_pbc)[1, 4] ≈ 0.25
        @test_throws ArgumentError ssh_chain(3; pbc=true)

        Haa = aubry_andre_chain(3; J=2.0, λ=0.5, β=0.25, ϕ=0.1, pbc=false)
        expected_aa = Matrix(hopping(3; J=2.0, pbc=false).h)
        expected_aa .+= diagm(ComplexF64[0.5 * cos(2π * 0.25 * j + 0.1) for j in 1:3])
        @test Matrix(Haa) ≈ expected_aa

        Hk = kitaev_chain(4; t=1.0, Δ=0.4, μ=0.2, pbc=false)
        A = zeros(ComplexF64, 4, 4)
        B = zeros(ComplexF64, 4, 4)
        for j in 1:4
            A[j, j] = -0.2
        end
        for j in 1:3
            A[j, j+1] = -1.0
            A[j+1, j] = -1.0
            B[j, j+1] = 0.4
            B[j+1, j] = -0.4
        end
        @test Matrix(Hk) ≈ Matrix(BdGHamiltonian(A, B))

        Hk_pbc = kitaev_chain(4; t=1.0, Δ=0.4, μ=0.2, pbc=true)
        A[4, 1] = -1.0
        A[1, 4] = -1.0
        B[4, 1] = 0.4
        B[1, 4] = -0.4
        @test Matrix(Hk_pbc) ≈ Matrix(BdGHamiltonian(A, B))

        @test_throws ArgumentError ssh_chain(0)
        @test_throws ArgumentError aubry_andre_chain(0)
        @test_throws ArgumentError kitaev_chain(0)
    end
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL with `ssh_chain` not defined.

- [ ] **Step 3: Create the model constructor implementation**

Create `src/Models.jl`:

```julia
#---------------------------------------------------------------------------------------------------
# Lightweight finite model constructors and spectral helpers
#---------------------------------------------------------------------------------------------------
export ssh_chain, aubry_andre_chain, kitaev_chain

function _check_model_length(L::Integer, name::String)
    L ≥ 1 || throw(ArgumentError("$name requires L ≥ 1, got $L"))
    Int(L)
end

"""
    ssh_chain(L; t1=1.0, t2=0.5, pbc=false) -> QuadraticHamiltonian

Finite Su-Schrieffer-Heeger chain with alternating nearest-neighbour hoppings.
Bond `(j, j+1)` uses `t1` for odd `j` and `t2` for even `j`. Periodic boundaries
require even `L` so the dimerization pattern closes consistently.
"""
function ssh_chain(L::Integer; t1::Number=1.0, t2::Number=0.5, pbc::Bool=false)
    L = _check_model_length(L, "ssh_chain")
    pbc && isodd(L) &&
        throw(ArgumentError("ssh_chain with pbc=true requires even L, got $L"))

    edges = Tuple{Int,Int,Number}[]
    for j in 1:L-1
        push!(edges, (j, j + 1, isodd(j) ? t1 : t2))
    end
    pbc && push!(edges, (L, 1, t2))
    hopping(edges, L)
end

"""
    aubry_andre_chain(L; J=1.0, λ=1.0, β=(sqrt(5)-1)/2, ϕ=0.0, pbc=false)

Finite Aubry-Andre chain with nearest-neighbour hopping `J` and onsite potential
`λ cos(2π β j + ϕ)`.
"""
function aubry_andre_chain(L::Integer; J::Number=1.0, λ::Real=1.0,
                           β::Real=(sqrt(5) - 1) / 2, ϕ::Real=0.0,
                           pbc::Bool=false)
    L = _check_model_length(L, "aubry_andre_chain")
    onsite = [λ * cos(2π * β * j + ϕ) for j in 1:L]
    hopping(L; J, pbc) + chemical_potential(onsite)
end

"""
    kitaev_chain(L; t=1.0, Δ=1.0, μ=0.0, pbc=false) -> BdGHamiltonian

Finite spinless p-wave superconducting chain in the package BdG block convention:
`A[j,j] = -μ`, nearest-neighbour hopping `-t`, and antisymmetric pairing block
`B[j,j+1] = Δ`.
"""
function kitaev_chain(L::Integer; t::Number=1.0, Δ::Number=1.0, μ::Real=0.0,
                      pbc::Bool=false)
    L = _check_model_length(L, "kitaev_chain")
    A = zeros(ComplexF64, L, L)
    B = zeros(ComplexF64, L, L)
    for j in 1:L
        A[j, j] = -μ
    end
    for j in 1:L-1
        A[j, j+1] = -t
        A[j+1, j] = -conj(t)
        B[j, j+1] = Δ
        B[j+1, j] = -Δ
    end
    if pbc && L > 1
        A[L, 1] = -t
        A[1, L] = -conj(t)
        B[L, 1] = Δ
        B[1, L] = -Δ
    end
    BdGHamiltonian(A, B)
end
```

- [ ] **Step 4: Include the new file**

Modify `src/GaussianFermions.jl` so the Hamiltonian area reads:

```julia
include("Hamiltonians.jl")  # QuadraticHamiltonian + propagator
include("Channels.jl")      # dissipation / measurement channels
include("Trajectory.jl")    # evolution, gates, quantum-trajectory primitives
include("CorrelationLindblad.jl")
# --- Majorana / BdG covariance layer ---
include("MajoranaStates.jl")
include("BdGHamiltonians.jl")
include("Models.jl")
include("MajoranaObservables.jl")
include("MajoranaLindblad.jl")
include("MajoranaTrajectory.jl")
```

- [ ] **Step 5: Run tests and verify constructor behavior passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: PASS for `"Model constructors"` and no regressions elsewhere.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/Models.jl src/GaussianFermions.jl test/runtests.jl
git commit -m "Add finite model constructors"
```

---

### Task 2: Spectral Helpers And Bloch Bands

**Files:**
- Modify: `src/Models.jl`
- Test: `test/runtests.jl`

- [ ] **Step 1: Write failing spectral helper tests**

Append this testset after `"Model constructors"` in `test/runtests.jl`:

```julia
    @testset "Spectral helpers" begin
        Hssh = ssh_chain(4; t1=1.0, t2=0.25)
        @test energy_spectrum(Hssh) ≈ sort(real.(eigvals(Hermitian(Matrix(Hssh)))))

        Hk = kitaev_chain(4; t=1.0, Δ=0.4, μ=0.2)
        @test energy_spectrum(Hk) ≈ quasiparticle_energies(Hk)

        Hbloch(k) = ComplexF64[
            0                  1 + exp(-im * k)
            1 + exp(im * k)    0
        ]
        bands = bloch_bands(Hbloch, [0.0, π])
        @test size(bands) == (2, 2)
        @test bands[1, :] ≈ [-2.0, 2.0]
        @test bands[2, :] ≈ [0.0, 0.0] atol = 1e-12

        bad_Hk(k) = k == 0 ? zeros(ComplexF64, 2, 2) : zeros(ComplexF64, 3, 3)
        @test_throws ArgumentError bloch_bands(bad_Hk, [0.0, 1.0])
    end
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL with `energy_spectrum` not defined.

- [ ] **Step 3: Add spectral helpers**

Append this implementation to `src/Models.jl`:

```julia
export energy_spectrum, bloch_bands

"""
    energy_spectrum(H::QuadraticHamiltonian) -> Vector{Float64}
    energy_spectrum(H::BdGHamiltonian) -> Vector{Float64}

Dense finite-system spectrum. For `QuadraticHamiltonian`, returns the single-particle
energies. For `BdGHamiltonian`, returns the non-negative quasiparticle energies.
"""
energy_spectrum(H::QuadraticHamiltonian) = sort(real.(eigvals(Hermitian(Matrix(H)))))
energy_spectrum(H::BdGHamiltonian) = quasiparticle_energies(H)

function _bloch_energy_values(H::AbstractMatrix)
    vals = eigvals(Hermitian(Matrix{ComplexF64}(H)))
    sort(real.(vals))
end
_bloch_energy_values(H::QuadraticHamiltonian) = energy_spectrum(H)
_bloch_energy_values(H::BdGHamiltonian) = energy_spectrum(H)

"""
    bloch_bands(Hk, kgrid) -> Matrix{Float64}

Evaluate a Bloch Hamiltonian function `Hk(k)` on `kgrid` and return a matrix whose
rows correspond to momenta and columns correspond to sorted bands. `Hk(k)` may
return a Hermitian matrix, `QuadraticHamiltonian`, or `BdGHamiltonian`.
"""
function bloch_bands(Hk, kgrid)
    ks = collect(kgrid)
    isempty(ks) && return zeros(Float64, 0, 0)
    values = [_bloch_energy_values(Hk(k)) for k in ks]
    nbands = length(first(values))
    all(v -> length(v) == nbands, values) ||
        throw(ArgumentError("bloch_bands requires Hk(k) to return the same number of bands for every k"))

    bands = zeros(Float64, length(ks), nbands)
    for (row, vals) in enumerate(values)
        bands[row, :] .= vals
    end
    bands
end
```

- [ ] **Step 4: Run tests and verify spectral behavior passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: PASS for `"Spectral helpers"` and no regressions elsewhere.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/Models.jl test/runtests.jl
git commit -m "Add finite spectral helpers"
```

---

### Task 3: Documentation And Example Updates

**Files:**
- Modify: `docs/src/reference/hamiltonians.md`
- Modify: `docs/src/manual/hamiltonians.md`
- Modify: `docs/src/examples/kitaev-chain.md`

- [ ] **Step 1: Update reference docs**

Change `docs/src/reference/hamiltonians.md` to:

````markdown
# Hamiltonians

```@docs
QuadraticHamiltonian
BdGHamiltonian
hopping
chemical_potential
ssh_chain
aubry_andre_chain
kitaev_chain
propagator
groundstate
quasiparticle_energies
energy_spectrum
bloch_bands
```
````

- [ ] **Step 2: Update the Hamiltonians manual**

Insert this section after the number-conserving Hamiltonian example in `docs/src/manual/hamiltonians.md`:

````markdown
## Finite Model Constructors

The package includes small finite constructors for common benchmarks. These are
not a symbolic lattice DSL: they return the same dense Hamiltonian types used by
the rest of the package.

```@example hamiltonians
ssh = ssh_chain(8; t1=1.0, t2=0.4)
aa = aubry_andre_chain(8; J=1.0, λ=0.8, β=(sqrt(5)-1)/2, ϕ=0.0)
kitaev = kitaev_chain(8; t=1.0, Δ=1.0, μ=0.5)

(ssh_lowest = round.(energy_spectrum(ssh)[1:2]; digits=4),
 kitaev_gap = round(first(energy_spectrum(kitaev)); digits=4))
```

For momentum-space workflows, pass a Bloch Hamiltonian function to `bloch_bands`:

```@example hamiltonians
Hk(k) = ComplexF64[
    0                  1 + exp(-im * k)
    1 + exp(im * k)    0
]
ks = range(-π, π; length=5)
bloch_bands(Hk, ks)
```
````

- [ ] **Step 3: Replace the local Kitaev helper in the example**

In `docs/src/examples/kitaev-chain.md`, replace the first code block that defines `kitaev_blocks` with:

````markdown
```@example kitaev
using GaussianFermions, LinearAlgebra
nothing # hide
```
````

Then replace:

```julia
Atop, Btop = kitaev_blocks(12; t=1.0, Δ=1.0, μ=0.0)   # topological
Atriv, Btriv = kitaev_blocks(12; t=1.0, Δ=1.0, μ=3.0) # trivial

εtop  = quasiparticle_energies(BdGHamiltonian(Atop, Btop))
εtriv = quasiparticle_energies(BdGHamiltonian(Atriv, Btriv))
```

with:

```julia
Htop  = kitaev_chain(12; t=1.0, Δ=1.0, μ=0.0)   # topological
Htriv = kitaev_chain(12; t=1.0, Δ=1.0, μ=3.0)   # trivial

εtop  = energy_spectrum(Htop)
εtriv = energy_spectrum(Htriv)
```

Replace:

```julia
A, B = kitaev_blocks(12; t=1.0, Δ=1.0, μ=0.5)   # detuned, still topological
H = BdGHamiltonian(A, B)
```

with:

```julia
H = kitaev_chain(12; t=1.0, Δ=1.0, μ=0.5)   # detuned, still topological
```

- [ ] **Step 4: Run docs doctests and package tests**

Run:

```bash
julia --project=. test/runtests.jl
julia --project=docs docs/make.jl
```

Expected: package tests pass and docs build without doctest failures.

- [ ] **Step 5: Commit Task 3**

```bash
git add docs/src/reference/hamiltonians.md docs/src/manual/hamiltonians.md docs/src/examples/kitaev-chain.md
git commit -m "Document model and spectral utilities"
```

---

### Task 4: Review Follow-On Scope

**Files:**
- Create: `docs/superpowers/specs/2026-06-04-stage-3-roadmap-notes.md`

- [ ] **Step 1: Create roadmap notes for deferred slices**

Create `docs/superpowers/specs/2026-06-04-stage-3-roadmap-notes.md`:

```markdown
# Stage 3 Roadmap Notes

## Confirmed Direction

GaussianFermions.jl should stay centered on finite fermionic Gaussian states under
closed, Lindblad, and monitored dynamics. Stage 3A adds lightweight model and
spectral helpers without changing that identity.

## Follow-On Slices

1. General observable engine:
   - `expect_bilinear`
   - `expect_quadratic`
   - `pfaffian_expectation`
   - scoped Wick helpers for Majorana products
   - optional full-counting statistics once the bilinear/Pfaffian API is stable

2. Optional scalability prototype:
   - benchmark current dense `:expm` paths
   - prototype `LinearMaps` plus Krylov action on vectorized covariance state
   - keep Krylov/SciML dependencies weak or extension-only

3. Topology:
   - start with one-dimensional winding/Pfaffian invariants tied to the new model helpers
   - defer full Chern/Z2 machinery or integrate with existing topology packages

4. Trajectory ensemble runner:
   - keep caller-owned loops as the default
   - only add a runner if it is explicitly transparent, optional, and example-backed

## Deferred As Separate Projects

- QuantumLattices-style symbolic model construction
- Kubo/transport response stack
- general Gaussian CP maps
- MPS/tensor-network conversion and compression
```

- [ ] **Step 2: Commit roadmap notes**

```bash
git add docs/superpowers/specs/2026-06-04-stage-3-roadmap-notes.md
git commit -m "Record Stage 3 follow-on scope"
```

---

## Self-Review

- **Spec coverage:** This plan implements the first near-term revision slice recommended by the verification pass: small model constructors, finite spectra, and simple Bloch-band evaluation. It records the other confirmed gaps as separate follow-on slices.
- **Placeholder scan:** No task uses `TBD`, unspecified validation, or references to undefined APIs outside the task where they are introduced.
- **Type consistency:** `ssh_chain` and `aubry_andre_chain` return `QuadraticHamiltonian`; `kitaev_chain` returns `BdGHamiltonian`; `energy_spectrum` dispatches on both; `bloch_bands` accepts matrices or either Hamiltonian type.

Plan complete and saved to `docs/superpowers/plans/2026-06-04-stage-3a-model-spectral-utilities.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using `superpowers:executing-plans`, with checkpoints.
