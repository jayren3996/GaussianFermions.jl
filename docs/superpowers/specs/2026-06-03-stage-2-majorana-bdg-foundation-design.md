# Stage 2 Design: Majorana/BdG Foundation

## Context

Stage 1a introduced the number-conserving Gaussian-state hierarchy:
`SlaterState`, `CorrelationState`, `QuadraticHamiltonian`, observables, trajectory
dynamics, no-click evolution, and the ensemble runner. Stage 1b added
`CorrelationLindblad` for deterministic number-conserving mixed-state dynamics and
steady states.

The remaining legacy Stage 2 code is concentrated in `src/Quadratic.jl`. It exposes
raw matrix helpers for Majorana covariance matrices and quadratic Lindblad evolution,
but it is not integrated with the Stage 1 API style. It also keeps older naming and
convention choices, including `covariancematrix`, `fermioncorrelation`,
`majoranaform`, and the misspelled `QuardraticLindblad`.

This pass is allowed to be aggressive and breaking. The goal is to replace the
legacy Majorana/BdG surface with a clean, typed foundation that matches the rest of
the package. Open-system Majorana Lindblad dynamics and BdG trajectories are left
for later Stage 2 slices.

## Goals

- Add canonical state and Hamiltonian types for general fermionic Gaussian states:
  `MajoranaState` and `BdGHamiltonian`.
- Introduce a top-level `AbstractGaussianState` so the existing number-conserving
  hierarchy and the new Majorana/BdG hierarchy share a public conceptual root.
- Make covariance, Dirac normal correlation, and anomalous correlation conventions
  explicit and testable.
- Provide dense unitary evolution for Majorana/BdG covariance states.
- Extend core observables (`density`, `particle_number`, entropies, mutual
  information) to `MajoranaState`.
- Replace the legacy Stage 2 helper names with clear public names.
- Keep the implementation focused enough that it can be verified before adding
  Lindblad or trajectory behavior.

## Non-Goals

- No rewrite of Majorana/BdG Lindblad dynamics in this slice.
- No BdG quantum-jump, diffusive, or no-click trajectory engine in this slice.
- No compatibility wrappers for the old `Quadratic.jl` names.
- No scalable Krylov/Faber/Schur propagator beyond dense `exp`.
- No attempt to merge `SlaterState` and `CorrelationState` into one universal state
  type. They remain the canonical number-conserving types.

## Public API

New or heavily revised files:

```julia
src/MajoranaStates.jl
src/BdGHamiltonians.jl
src/MajoranaObservables.jl
```

`src/Quadratic.jl` should either be deleted or reduced to implementation-private
helpers that do not export legacy names. The canonical public API should come from
the new files.

Exports:

```julia
AbstractGaussianState
MajoranaState
BdGHamiltonian
covariance_matrix
normal_correlation
anomalous_correlation
fermion_correlations
```

Existing exported names extended to `MajoranaState`:

```julia
nmodes
evolve!
evolve
density
particle_number
entanglement_entropy
renyi_entropy
mutual_information
tripartite_information
entanglement_spectrum
```

Legacy names intentionally removed from the public API:

```julia
covariancematrix
fermioncorrelation
majoranaform
quadraticlindblad
quadraticlindblad_from_fermion
fermionlindblad
lindblad_evo
```

## State Convention

Use a real Majorana basis ordered as

```text
gamma = [x_1, ..., x_L, p_1, ..., p_L]
```

The covariance matrix is

```text
Gamma[i,j] = (i/2) * <[gamma_i, gamma_j]>
```

For every `MajoranaState`, `Gamma` is a real `2L x 2L` antisymmetric matrix. Physical
states have covariance singular values no larger than one, equivalently the
eigenvalues of `im * Gamma` lie in `[-1, 1]` up to numerical tolerance.

The new `MajoranaState(occ)` constructor should preserve the old product-state
meaning:

```text
occ_i = 1 -> site i occupied
occ_i = 0 -> site i empty
```

For this basis, a product occupation vector maps to diagonal `x/p` blocks with
entries `1 - 2 * occ_i` in the upper-right block and the opposite sign in the
lower-left block. This keeps compatibility with the old `covariancematrix`
mathematics without keeping its name.

## State Types

Add a top-level state interface:

```julia
abstract type AbstractGaussianState{T<:Number} end
```

Change the existing number-conserving root to:

```julia
abstract type NumberConservingGaussianState{T<:Number} <: AbstractGaussianState{T} end
```

Add:

```julia
mutable struct MajoranaState{T<:Real} <: AbstractGaussianState{T}
    Gamma::Matrix{T}
end
```

Constructors:

```julia
MajoranaState(Gamma::AbstractMatrix; check=true, atol=1e-10)
MajoranaState(occ::AbstractVector{<:Integer})
MajoranaState(s::SlaterState)
MajoranaState(s::CorrelationState; check=true)
```

Validation when `check=true`:

- `Gamma` is square.
- `size(Gamma, 1)` is even.
- all entries are finite and real-valued.
- `Gamma` is antisymmetric within `atol`.
- the physical covariance spectrum is inside `[-1, 1]` within tolerance.

Core accessors:

```julia
nmodes(s::MajoranaState)
covariance_matrix(s::MajoranaState)
Base.copy(s::MajoranaState)
Base.eltype(::MajoranaState{T}) where {T}
```

`covariance_matrix` returns a copy, not the internal mutable matrix.

## Fermion Correlations

Provide explicit Dirac-correlation accessors:

```julia
fermion_correlations(s::MajoranaState) -> (C, F)
normal_correlation(s::MajoranaState) -> Matrix{ComplexF64}
anomalous_correlation(s::MajoranaState) -> Matrix{ComplexF64}
```

Definitions:

- `C[i,j] = <c_i^+ c_j>`.
- `F[i,j] = <c_i c_j>`.

`density(s::MajoranaState)` is `real(diag(normal_correlation(s)))`.

For converted number-conserving states:

- `normal_correlation(MajoranaState(s))` agrees with `correlation_matrix(s)`.
- `anomalous_correlation(MajoranaState(s))` is zero within numerical tolerance.

## Hamiltonian Convention

Use a real antisymmetric Majorana generator for time evolution:

```julia
mutable struct BdGHamiltonian{T<:Real} <: AbstractQuadraticHamiltonian
    K::Matrix{T}
    cache::Union{Nothing,Tuple{Float64,Matrix{T}}}
end
```

`K` is the generator for covariance evolution:

```text
O(dt) = exp(K * dt)
Gamma(dt) = O(dt) * Gamma(0) * transpose(O(dt))
```

Constructors:

```julia
BdGHamiltonian(K::AbstractMatrix; check=true, atol=1e-10)
BdGHamiltonian(A::AbstractMatrix, B::AbstractMatrix; check=true, atol=1e-10)
BdGHamiltonian(H::QuadraticHamiltonian)
```

`A` and `B` are Dirac BdG blocks in the convention already documented in the legacy
code:

```text
Hhat = 1/2 * sum_ij (
    A_ij c_i^+ c_j + B_ij c_i^+ c_j^+ + h.c.
)
```

`A` must be Hermitian and `B` must be antisymmetric within tolerance. The constructor
uses the legacy `majoranaform(A, B)` algebra internally to build the Hamiltonian-form
antisymmetric matrix `Hmaj` for

```text
Hhat = -im/4 * sum_ij Hmaj_ij gamma_i gamma_j
```

and stores the covariance evolution generator `K = -Hmaj`. This sign is part of the
public convention: `BdGHamiltonian` stores the generator used in
`Gamma(dt) = O * Gamma(0) * transpose(O)`, not the raw Hamiltonian-form matrix.

For a number-conserving `QuadraticHamiltonian`, `BdGHamiltonian(H)` must produce
Majorana evolution that agrees with `evolve!(::CorrelationState, H, dt)` after
converting back through `normal_correlation`.

Hamiltonian accessors:

```julia
nmodes(H::BdGHamiltonian)
Base.Matrix(H::BdGHamiltonian)
propagator(H::BdGHamiltonian, dt::Real)
```

`propagator` returns the cached orthogonal matrix `O = exp(K * dt)`.

## Evolution

Extend the existing evolution verbs:

```julia
evolve!(s::MajoranaState, O::AbstractMatrix)
evolve!(s::MajoranaState, H::BdGHamiltonian, dt::Real)
evolve(s::MajoranaState, args...)
```

Rules:

- `evolve!(s, O)` applies `Gamma -> O * Gamma * transpose(O)`.
- It validates size compatibility.
- It re-antisymmetrizes roundoff with `(Gamma - transpose(Gamma)) / 2`.
- It does not silently clamp physical eigenvalues.
- `evolve(s, ...)` returns a copy.

## Observables

Extend existing observable names rather than creating a separate Majorana vocabulary.

For `MajoranaState`:

```julia
density(s)
particle_number(s)
particle_number(s, A)
entanglement_entropy(s, A)
renyi_entropy(s, A; α=2)
mutual_information(s, A, B; α=1)
tripartite_information(s, A, B, C; α=1)
entanglement_spectrum(s, A)
```

`particle_number(s)` is defined as `sum(density(s))`; it is an expectation value, not
a conserved charge guarantee.

For entanglement, build the restricted Majorana covariance block using the region
ordering `[A; A .+ L]`, compute the positive covariance singular values, and use

```text
S = -sum_k [
    ((1 + nu_k)/2) * log((1 + nu_k)/2)
  + ((1 - nu_k)/2) * log((1 - nu_k)/2)
]
```

The same spectrum should power Renyi entropies and mutual information. For
number-conserving converted states, `entanglement_entropy(MajoranaState(s), A)` must
agree with `entanglement_entropy(s, A)`.

## File Integration

Update `src/GaussianFermions.jl` include order to keep the old Stage 1 dependency
flow and then load Stage 2:

```text
LinAlg -> Modes -> States -> Observables -> Hamiltonians -> Channels ->
Trajectory -> CorrelationLindblad -> MajoranaStates -> BdGHamiltonians ->
MajoranaObservables
```

If any small linear-algebra helpers from `Quadratic.jl` remain useful, keep them
unexported in the new files or a private helper file. Do not keep `Quadratic.jl` as
a public legacy module.

## Tests

Add a dedicated `@testset "Majorana/BdG foundation"` to `test/runtests.jl`.

Required tests:

- `MajoranaState([1, 0, 1, 0])` has four modes, a real antisymmetric `8 x 8`
  covariance matrix, density `[1, 0, 1, 0]`, and zero entropy for product-state
  subregions that are unentangled.
- `MajoranaState(SlaterState(...))` round-trips normal correlations:
  `normal_correlation(ms) ≈ correlation_matrix(slater)` and
  `anomalous_correlation(ms) ≈ 0`.
- `MajoranaState(CorrelationState(...))` supports mixed states, and its density
  matches the input correlation state.
- invalid covariance inputs throw `ArgumentError`: odd size, non-square,
  non-antisymmetric, non-finite, or unphysical spectrum.
- `BdGHamiltonian(K)` rejects non-square, odd-size, non-antisymmetric, or non-finite
  generators.
- `propagator(BdGHamiltonian(K), dt)` is orthogonal and cached for repeated `dt`.
- Number-conserving agreement: evolve a `CorrelationState` with `QuadraticHamiltonian`
  and evolve the converted `MajoranaState` with `BdGHamiltonian(H)`, then compare
  `normal_correlation`.
- Pairing behavior: a valid antisymmetric pairing block `B` produces nonzero
  `anomalous_correlation` after evolution from a product state.
- Entanglement agreement: for a number-conserving Slater state, Majorana and
  number-conserving entanglement entropy agree on the same region.

## Documentation

Update `README.md` with a short Majorana/BdG example after implementation:

```julia
using GaussianFermions

s = MajoranaState([1, 0, 1, 0])
A = zeros(ComplexF64, 4, 4)
B = zeros(ComplexF64, 4, 4)
B[1, 2] = 0.3
B[2, 1] = -0.3
H = BdGHamiltonian(A, B)

evolve!(s, H, 0.5)
normal_correlation(s)
anomalous_correlation(s)
```

The README should state that Majorana/BdG Lindblad and trajectory support are planned
later Stage 2 slices.

## Migration Notes

This is a breaking Stage 2 cleanup:

- Use `MajoranaState(occ)` instead of `covariancematrix(occ)`.
- Use `covariance_matrix(s)` instead of passing raw `Gamma` matrices around.
- Use `normal_correlation(s)` and `anomalous_correlation(s)` instead of
  `fermioncorrelation(Gamma, selector)`.
- Use `BdGHamiltonian(A, B)` or `BdGHamiltonian(H)` instead of `majoranaform`.
- Majorana/BdG Lindblad APIs are intentionally absent until the next Stage 2 slice.
