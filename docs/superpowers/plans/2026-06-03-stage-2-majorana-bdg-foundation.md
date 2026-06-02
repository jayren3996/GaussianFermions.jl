# Stage 2 Majorana/BdG Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy raw `Quadratic.jl` Stage 2 surface with typed `MajoranaState` and `BdGHamiltonian` APIs, dense unitary evolution, Dirac correlation accessors, and Majorana observables.

**Architecture:** Add a top-level `AbstractGaussianState`, keep the existing number-conserving hierarchy under it, and add focused Stage 2 files loaded after Stage 1/1b. `MajoranaState` stores real antisymmetric covariance matrices; `BdGHamiltonian` stores the covariance evolution generator `K`; observables dispatch on `MajoranaState` through covariance spectra and Dirac correlations.

**Tech Stack:** Julia, LinearAlgebra dense matrix exponential/eigendecomposition, existing `Test` suite, existing `QuadraticHamiltonian`/`CorrelationState` APIs.

---

## File Structure

- Modify `src/States.jl`: introduce `AbstractGaussianState`; make `NumberConservingGaussianState` subtype it.
- Create `src/MajoranaStates.jl`: `MajoranaState`, validation, covariance accessors, conversions from occupation vectors, `SlaterState`, and `CorrelationState`, and Dirac correlation accessors.
- Create `src/BdGHamiltonians.jl`: `BdGHamiltonian`, Dirac block conversion, validation, propagator caching, and `evolve!`/`evolve` methods.
- Create `src/MajoranaObservables.jl`: `density`, `particle_number`, covariance entanglement spectrum/entropy/Renyi/mutual/tripartite methods for `MajoranaState`.
- Modify `src/GaussianFermions.jl`: remove legacy `Quadratic.jl` include and `Base.:*` import; include/export new Stage 2 files.
- Delete `src/Quadratic.jl`: old public Stage 2 API is intentionally removed.
- Modify `test/runtests.jl`: add focused `Majorana/BdG foundation` tests.
- Modify `README.md`: add a short Majorana/BdG foundation example and note that Majorana Lindblad/trajectories are later slices.

---

### Task 1: Majorana State And Correlation Accessors

**Files:**
- Modify: `src/States.jl`
- Create: `src/MajoranaStates.jl`
- Modify: `src/GaussianFermions.jl`
- Test: `test/runtests.jl`

- [ ] **Step 1: Write failing state/conversion tests**

Add this testset after `"Mixed states & conversions"` in `test/runtests.jl`:

```julia
    @testset "Majorana/BdG foundation states" begin
        occ = [1, 0, 1, 0]
        ms = MajoranaState(occ)
        Γ = covariance_matrix(ms)
        @test nmodes(ms) == 4
        @test size(Γ) == (8, 8)
        @test eltype(Γ) == Float64
        @test Γ ≈ -transpose(Γ)
        @test normal_correlation(ms) ≈ diagm(ComplexF64.(occ))
        @test anomalous_correlation(ms) ≈ zeros(ComplexF64, 4, 4)

        slater = SlaterState(L=4, N=2, config="Z2")
        from_slater = MajoranaState(slater)
        @test normal_correlation(from_slater) ≈ correlation_matrix(slater)
        @test anomalous_correlation(from_slater) ≈ zeros(ComplexF64, 4, 4) atol = 1e-12

        cmat = ComplexF64[
            0.6   0.2im
           -0.2im 0.4
        ]
        corr = CorrelationState(cmat)
        from_corr = MajoranaState(corr)
        @test normal_correlation(from_corr) ≈ correlation_matrix(corr)

        @test covariance_matrix(ms) !== ms.Gamma
        Γcopy = covariance_matrix(ms)
        Γcopy[1, 2] = 99
        @test covariance_matrix(ms)[1, 2] != 99

        @test_throws ArgumentError MajoranaState(zeros(3, 3))
        @test_throws ArgumentError MajoranaState(zeros(2, 3))
        bad = zeros(4, 4); bad[1, 2] = 0.1
        @test_throws ArgumentError MajoranaState(bad)
        nonfinite = zeros(4, 4); nonfinite[1, 2] = Inf; nonfinite[2, 1] = -Inf
        @test_throws ArgumentError MajoranaState(nonfinite)
        unphysical = zeros(4, 4); unphysical[1, 3] = 2.0; unphysical[3, 1] = -2.0
        @test_throws ArgumentError MajoranaState(unphysical)
    end
```

- [ ] **Step 2: Run the new testset and verify it fails**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL because `MajoranaState` and `covariance_matrix` are not defined.

- [ ] **Step 3: Implement state hierarchy and `MajoranaState`**

In `src/States.jl`, replace the abstract type declaration with:

```julia
export AbstractGaussianState

abstract type AbstractGaussianState{T<:Number} end
abstract type NumberConservingGaussianState{T<:Number} <: AbstractGaussianState{T} end
```

Create `src/MajoranaStates.jl` with:

```julia
#---------------------------------------------------------------------------------------------------
# Majorana / BdG covariance states
#---------------------------------------------------------------------------------------------------
export MajoranaState, covariance_matrix
export fermion_correlations, normal_correlation, anomalous_correlation

mutable struct MajoranaState{T<:Real} <: AbstractGaussianState{T}
    Gamma::Matrix{T}
end

function MajoranaState(Gamma::AbstractMatrix; check::Bool=true, atol::Real=1e-10)
    size(Gamma, 1) == size(Gamma, 2) ||
        throw(ArgumentError("Majorana covariance matrix must be square, got $(size(Gamma))"))
    iseven(size(Gamma, 1)) ||
        throw(ArgumentError("Majorana covariance matrix size must be even, got $(size(Gamma, 1))"))
    G = Matrix{Float64}(Gamma)
    all(isfinite, G) ||
        throw(ArgumentError("Majorana covariance matrix entries must be finite"))
    if check
        isapprox(G, -transpose(G); atol) ||
            throw(ArgumentError("Majorana covariance matrix must be antisymmetric"))
        vals = real.(eigvals(Hermitian(1im * G)))
        maximum(abs.(vals); init=0.0) ≤ 1 + sqrt(atol) ||
            throw(ArgumentError("Majorana covariance matrix has unphysical spectrum outside [-1, 1]"))
    end
    MajoranaState{Float64}((G - transpose(G)) / 2)
end

function MajoranaState(occ::AbstractVector{<:Integer})
    all(x -> x == 0 || x == 1, occ) ||
        throw(ArgumentError("occupation vector entries must be 0 or 1"))
    L = length(occ)
    D = Diagonal(1 .- 2 .* Float64.(occ))
    Z = zeros(Float64, L, L)
    MajoranaState([Z D; -D Z]; check=false)
end

MajoranaState(s::SlaterState) = MajoranaState(CorrelationState(s))

function MajoranaState(s::CorrelationState; check::Bool=true, atol::Real=1e-10)
    C = correlation_matrix(s)
    L = size(C, 1)
    X = 2 .* imag.(C)
    D = I - 2 .* real.(C)
    G = Matrix{Float64}([X D; -D X])
    MajoranaState(G; check, atol)
end

nmodes(s::MajoranaState) = size(s.Gamma, 1) ÷ 2
Base.eltype(::MajoranaState{T}) where {T} = T
Base.copy(s::MajoranaState) = MajoranaState(copy(s.Gamma); check=false)
covariance_matrix(s::MajoranaState) = copy(s.Gamma)

function fermion_correlations(s::MajoranaState)
    G = s.Gamma
    L = nmodes(s)
    G11 = view(G, 1:L, 1:L)
    G12 = view(G, 1:L, L+1:2L)
    G21 = view(G, L+1:2L, 1:L)
    G22 = view(G, L+1:2L, L+1:2L)
    C = (G21 .- G12 .+ 1im .* G11 .+ 1im .* G22) ./ 4 .+ I / 2
    F = (G21 .+ G12 .+ 1im .* G11 .- 1im .* G22) ./ 4
    Matrix{ComplexF64}(C), Matrix{ComplexF64}(F)
end

normal_correlation(s::MajoranaState) = first(fermion_correlations(s))
anomalous_correlation(s::MajoranaState) = last(fermion_correlations(s))
```

Modify `src/GaussianFermions.jl` to include `MajoranaStates.jl` after `CorrelationLindblad.jl`. Keep `Quadratic.jl` included until Task 4 removes it.

- [ ] **Step 4: Run tests and verify Task 1 passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: all existing tests and the new state tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/States.jl src/MajoranaStates.jl src/GaussianFermions.jl test/runtests.jl
git commit -m "Add Majorana covariance state"
```

---

### Task 2: BdG Hamiltonian And Majorana Evolution

**Files:**
- Create: `src/BdGHamiltonians.jl`
- Modify: `src/GaussianFermions.jl`
- Test: `test/runtests.jl`

- [ ] **Step 1: Write failing Hamiltonian/evolution tests**

Append these tests inside `"Majorana/BdG foundation states"` or rename that testset to `"Majorana/BdG foundation"`:

```julia
        K = zeros(Float64, 4, 4)
        K[1, 3] = 0.7; K[3, 1] = -0.7
        bdg = BdGHamiltonian(K)
        O = propagator(bdg, 0.2)
        @test O * transpose(O) ≈ I(4) atol = 1e-12
        @test propagator(bdg, 0.2) === propagator(bdg, 0.2)
        @test_throws ArgumentError BdGHamiltonian(zeros(3, 3))
        @test_throws ArgumentError BdGHamiltonian(zeros(2, 3))
        notanti = zeros(4, 4); notanti[1, 2] = 0.1
        @test_throws ArgumentError BdGHamiltonian(notanti)
        badK = zeros(4, 4); badK[1, 2] = NaN; badK[2, 1] = -NaN
        @test_throws ArgumentError BdGHamiltonian(badK)

        Hnc = hopping(4; pbc=true)
        cstate = CorrelationState(SlaterState(L=4, N=2, config="Z2"))
        mstate = MajoranaState(cstate)
        evolve!(cstate, Hnc, 0.37)
        evolve!(mstate, BdGHamiltonian(Hnc), 0.37)
        @test normal_correlation(mstate) ≈ correlation_matrix(cstate) atol = 1e-10

        evolved_copy = evolve(mstate, BdGHamiltonian(Hnc), 0.1)
        @test evolved_copy !== mstate
        @test covariance_matrix(evolved_copy) != covariance_matrix(mstate)

        @test_throws ArgumentError evolve!(MajoranaState([1, 0]), Matrix(I, 6, 6))
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL because `BdGHamiltonian` is not defined.

- [ ] **Step 3: Implement `BdGHamiltonian` and evolution**

Create `src/BdGHamiltonians.jl`:

```julia
#---------------------------------------------------------------------------------------------------
# BdG / Majorana quadratic Hamiltonians
#---------------------------------------------------------------------------------------------------
export BdGHamiltonian

mutable struct BdGHamiltonian{T<:Real} <: AbstractQuadraticHamiltonian
    K::Matrix{T}
    cache::Union{Nothing,Tuple{Float64,Matrix{T}}}
end

function _check_square_even_matrix(M::AbstractMatrix, label::String)
    size(M, 1) == size(M, 2) ||
        throw(ArgumentError("$label must be square, got $(size(M))"))
    iseven(size(M, 1)) ||
        throw(ArgumentError("$label size must be even, got $(size(M, 1))"))
end

function BdGHamiltonian(K::AbstractMatrix; check::Bool=true, atol::Real=1e-10)
    _check_square_even_matrix(K, "BdG Majorana generator")
    Kmat = Matrix{Float64}(K)
    all(isfinite, Kmat) ||
        throw(ArgumentError("BdG Majorana generator entries must be finite"))
    if check
        isapprox(Kmat, -transpose(Kmat); atol) ||
            throw(ArgumentError("BdG Majorana generator must be antisymmetric"))
    end
    BdGHamiltonian{Float64}((Kmat - transpose(Kmat)) / 2, nothing)
end

function _majorana_hamiltonian_matrix(A::AbstractMatrix, B::AbstractMatrix)
    size(A) == size(B) ||
        throw(ArgumentError("BdG blocks A and B must have the same size, got $(size(A)) and $(size(B))"))
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("BdG block A must be square, got $(size(A))"))
    isapprox(A, A'; atol=1e-10) ||
        throw(ArgumentError("BdG block A must be Hermitian"))
    isapprox(B, -transpose(B); atol=1e-10) ||
        throw(ArgumentError("BdG pairing block B must be antisymmetric"))
    AR, AI = real.(A), imag.(A)
    BR, BI = real.(B), imag.(B)
    Matrix{Float64}([-AI - BI AR - BR; -AR - BR -AI + BI])
end

BdGHamiltonian(A::AbstractMatrix, B::AbstractMatrix; check::Bool=true, atol::Real=1e-10) =
    BdGHamiltonian(-_majorana_hamiltonian_matrix(A, B); check, atol)

BdGHamiltonian(H::QuadraticHamiltonian) =
    BdGHamiltonian(-_majorana_hamiltonian_matrix(Matrix(H.h), zeros(ComplexF64, nmodes(H), nmodes(H))); check=false)

nmodes(H::BdGHamiltonian) = size(H.K, 1) ÷ 2
Base.Matrix(H::BdGHamiltonian) = copy(H.K)

function propagator(H::BdGHamiltonian, dt::Real)
    if H.cache !== nothing && H.cache[1] == Float64(dt)
        return H.cache[2]
    end
    O = exp(Float64(dt) .* H.K)
    H.cache = (Float64(dt), O)
    O
end

function evolve!(s::MajoranaState, O::AbstractMatrix)
    size(O) == size(s.Gamma) ||
        throw(ArgumentError("Majorana propagator size $(size(O)) does not match covariance size $(size(s.Gamma))"))
    Omat = Matrix{Float64}(O)
    G = Omat * s.Gamma * transpose(Omat)
    s.Gamma = (G - transpose(G)) / 2
    s
end

evolve!(s::MajoranaState, H::BdGHamiltonian, dt::Real) = evolve!(s, propagator(H, dt))
evolve(s::MajoranaState, args...) = evolve!(copy(s), args...)
```

Modify `src/GaussianFermions.jl` to include `BdGHamiltonians.jl` after `MajoranaStates.jl`.

- [ ] **Step 4: Run tests and verify Task 2 passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/BdGHamiltonians.jl src/GaussianFermions.jl test/runtests.jl
git commit -m "Add BdG Hamiltonian evolution"
```

---

### Task 3: Majorana Observables And Entanglement

**Files:**
- Create: `src/MajoranaObservables.jl`
- Modify: `src/GaussianFermions.jl`
- Test: `test/runtests.jl`

- [ ] **Step 1: Write failing observable tests**

Append these tests inside `"Majorana/BdG foundation"`:

```julia
        prod_state = MajoranaState([1, 0, 1, 0])
        @test density(prod_state) ≈ [1, 0, 1, 0]
        @test particle_number(prod_state) ≈ 2
        @test particle_number(prod_state, [1, 2]) ≈ 1
        @test entanglement_entropy(prod_state, [1, 2]) ≈ 0 atol = 1e-12
        @test renyi_entropy(prod_state, [1, 2]; α=2) ≈ 0 atol = 1e-12
        @test mutual_information(prod_state, [1], [2]) ≈ 0 atol = 1e-12
        @test tripartite_information(prod_state, [1], [2], [3]) ≈ 0 atol = 1e-12
        @test entanglement_spectrum(prod_state, [1, 2]) ≈ [1, 1] atol = 1e-12

        ent_slater = SlaterState([1], 2)
        evolve!(ent_slater, ComplexF64[1 1; 1 -1] / sqrt(2))
        ent_majorana = MajoranaState(ent_slater)
        @test density(ent_majorana) ≈ density(ent_slater)
        @test entanglement_entropy(ent_majorana, [1]) ≈ entanglement_entropy(ent_slater, [1]) atol = 1e-10
        @test renyi_entropy(ent_majorana, [1]; α=Inf) ≈ renyi_entropy(ent_slater, [1]; α=Inf) atol = 1e-10
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL because `particle_number(::MajoranaState)` and entropy methods are not defined.

- [ ] **Step 3: Implement Majorana observable methods**

Create `src/MajoranaObservables.jl`:

```julia
#---------------------------------------------------------------------------------------------------
# Majorana / BdG observables
#---------------------------------------------------------------------------------------------------

density(s::MajoranaState) = real.(diag(normal_correlation(s)))
density(s::MajoranaState, i::Integer) = density(s)[i]
particle_number(s::MajoranaState) = sum(density(s))
particle_number(s::MajoranaState, A) = sum(density(s, i) for i in A)

function _majorana_region_covariance(s::MajoranaState, A::AbstractVector{<:Integer})
    L = nmodes(s)
    inds = vcat(collect(A), collect(A) .+ L)
    covariance_matrix(s)[inds, inds]
end

function entanglement_spectrum(s::MajoranaState, A::AbstractVector{<:Integer})
    G = _majorana_region_covariance(s, A)
    n = length(A)
    vals = sort(real.(eigvals(Hermitian(1im * G))))
    clamp.(vals[n+1:end], 0.0, 1.0)
end

function entanglement_entropy(s::MajoranaState, A::AbstractVector{<:Integer})
    sum(entanglement_spectrum(s, A)) do ν
        _binary_shannon((1 + ν) / 2)
    end
end

function renyi_entropy(s::MajoranaState, A::AbstractVector{<:Integer}; α::Real=2)
    α == 1 && return entanglement_entropy(s, A)
    ν = entanglement_spectrum(s, A)
    p = @. (1 + ν) / 2
    isinf(α) && return -sum(log(max(pk, 1 - pk)) for pk in p)
    (1 / (1 - α)) * sum(log(pk^α + (1 - pk)^α) for pk in p)
end

function mutual_information(s::MajoranaState,
                            A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer}; α::Real=1)
    S(R) = renyi_entropy(s, R; α)
    S(A) + S(B) - S(vcat(A, B))
end

function tripartite_information(s::MajoranaState,
                                A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer},
                                C::AbstractVector{<:Integer}; α::Real=1)
    S(R) = renyi_entropy(s, R; α)
    S(A) + S(B) + S(C) - S(vcat(A, B)) - S(vcat(A, C)) - S(vcat(B, C)) + S(vcat(A, B, C))
end
```

Modify `src/GaussianFermions.jl` to include `MajoranaObservables.jl` after `BdGHamiltonians.jl`.

- [ ] **Step 4: Run tests and verify Task 3 passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/MajoranaObservables.jl src/GaussianFermions.jl test/runtests.jl
git commit -m "Add Majorana observables"
```

---

### Task 4: Pairing Behavior And Legacy API Removal

**Files:**
- Modify: `src/GaussianFermions.jl`
- Delete: `src/Quadratic.jl`
- Modify: `test/runtests.jl`

- [ ] **Step 1: Write failing pairing/removal tests**

Append these tests inside `"Majorana/BdG foundation"`:

```julia
        A = zeros(ComplexF64, 2, 2)
        B = zeros(ComplexF64, 2, 2)
        B[1, 2] = 0.5
        B[2, 1] = -0.5
        paired = MajoranaState([0, 0])
        evolve!(paired, BdGHamiltonian(A, B), 0.4)
        @test norm(anomalous_correlation(paired)) > 1e-3
        @test_throws ArgumentError BdGHamiltonian(A, ComplexF64[0 1; 1 0])

        @test !isdefined(GaussianFermions, :covariancematrix)
        @test !isdefined(GaussianFermions, :fermioncorrelation)
        @test !isdefined(GaussianFermions, :majoranaform)
        @test !isdefined(GaussianFermions, :quadraticlindblad)
        @test !isdefined(GaussianFermions, :lindblad_evo)
```

- [ ] **Step 2: Run tests and verify expected failure**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL while `Quadratic.jl` is still included and legacy names are still defined.

- [ ] **Step 3: Remove legacy Stage 2 include and file**

Modify `src/GaussianFermions.jl`:

```julia
module GaussianFermions

using LinearAlgebra, StaticArrays, LoopVectorization, Random

# --- number-conserving (U(1)) layer ---
include("LinAlg.jl")
include("Modes.jl")
include("States.jl")
include("Observables.jl")
include("Hamiltonians.jl")
include("Channels.jl")
include("Trajectory.jl")
include("CorrelationLindblad.jl")

# --- Majorana / BdG covariance layer ---
include("MajoranaStates.jl")
include("BdGHamiltonians.jl")
include("MajoranaObservables.jl")

end # module GaussianFermions
```

Delete `src/Quadratic.jl`.

- [ ] **Step 4: Run tests and verify Task 4 passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/GaussianFermions.jl src/Quadratic.jl test/runtests.jl
git commit -m "Remove legacy Quadratic API"
```

---

### Task 5: README And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Add this section after the deterministic correlation-matrix Lindblad examples:

````markdown
## Majorana/BdG Foundation

General quadratic (BdG/Majorana) Gaussian states use `MajoranaState`, which stores
the real antisymmetric Majorana covariance matrix. `BdGHamiltonian` evolves this
covariance under dense unitary dynamics.

```julia
using GaussianFermions

state = MajoranaState([1, 0, 1, 0])
A = zeros(ComplexF64, 4, 4)
B = zeros(ComplexF64, 4, 4)
B[1, 2] = 0.3
B[2, 1] = -0.3

H = BdGHamiltonian(A, B)
evolve!(state, H, 0.5)

normal_correlation(state)
anomalous_correlation(state)
```

Majorana/BdG Lindblad dynamics and trajectory support are planned as later Stage 2
slices.
````

- [ ] **Step 2: Run full verification**

Run:

```bash
julia --project=. test/runtests.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: both pass.

- [ ] **Step 3: Commit README**

```bash
git add README.md
git commit -m "Document Majorana BdG foundation"
```

- [ ] **Step 4: Final source scan**

Run:

```bash
rg -n "covariancematrix|fermioncorrelation|majoranaform|Quardratic|quadraticlindblad|lindblad_evo|include\\(\"Quadratic" src test README.md docs/superpowers/specs/2026-06-03-stage-2-majorana-bdg-foundation-design.md
git status --short --branch
```

Expected: old names appear only in the design spec migration/context notes, not in `src`, `test`, or `README.md`; git status is clean after commits.
