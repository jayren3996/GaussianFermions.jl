# Stage 1b Design: Correlation-Matrix Lindblad Dynamics

## Context

Claude completed Stage 1a on branch `stage1-overhaul`: the package now has
`SlaterState`, `CorrelationState`, observables, Hamiltonians, channels, trajectories,
no-click evolution, and an ensemble runner. The explicit deferred item is Stage 1b:
deterministic number-conserving mixed-state dynamics on `CorrelationState` plus a
steady-state solver.

This pass does not change `Quadratic.jl` or introduce Stage 2 Majorana/BdG state
types. It adds a number-conserving deterministic layer beside the existing trajectory
layer.

## Goals

- Add a public `CorrelationLindblad` type for deterministic Gaussian master equations
  on `CorrelationState`.
- Expose a low-level constructor for arbitrary linear loss/gain bath modes.
- Provide channel adapters for the existing `Loss`, `Gain`, `OccupationMonitor`, and
  `HoleMonitor` types.
- Add `lindblad_rhs`, `evolve!`, `evolve`, and `steadystate` APIs.
- Keep the implementation mathematically explicit and easy to test before optimizing
  large-system performance.

## Non-Goals

- No Stage 2 Majorana/BdG hierarchy work.
- No rewrite of `Quadratic.jl`.
- No distributed or Krylov time evolution in the first implementation. The initial
  solver can be dense and exact for finite systems; scalable variants can be added
  later without changing the public API.
- No attempt to make nonlinear or interacting Lindblad equations Gaussian. Supported
  deterministic dissipators are the Gaussian-preserving number-conserving ones listed
  here.

## Public API

New file:

```julia
src/CorrelationLindblad.jl
```

Exports:

```julia
CorrelationLindblad
lindblad_rhs
steadystate
```

`evolve!` and `evolve` are already exported by `Trajectory.jl`; Stage 1b extends them
for `CorrelationState` with `CorrelationLindblad`.

Primary constructors:

```julia
CorrelationLindblad(H::QuadraticHamiltonian; loss_ops=[], gain_ops=[], dephasing_ops=[])
CorrelationLindblad(h::AbstractMatrix; loss_ops=[], gain_ops=[], dephasing_ops=[])
CorrelationLindblad(H::QuadraticHamiltonian, channels)
CorrelationLindblad(h::AbstractMatrix, channels)
```

Low-level linear bath modes are vectors in the same single-particle mode convention
as `QuasiMode`: a normalized vector `v` represents `d+ = sum_i v_i c_i+`. For
`loss_ops` and `gain_ops`, rates are baked into the vector norm: `sqrt(gamma) * v`.

Dephasing is quadratic (`d+ d`), so its rate cannot be safely inferred from a
preweighted vector without ambiguity. Low-level dephasing entries use one of:

```julia
(v, gamma)          # mode vector plus rate
(P, gamma)          # single-particle projector plus rate
```

Channel adapters convert:

- `Loss(mode; gamma)` to one weighted loss vector.
- `Gain(mode; gamma)` to one weighted gain vector.
- `OccupationMonitor(mode; gamma)` to one dephasing projector.
- `HoleMonitor(mode; gamma)` to the same dephasing dissipator as occupation monitoring.

## State Convention

The package uses

```julia
C[i,j] = <c_i+ c_j>
C = conj(B) * transpose(B)
```

Under a single-particle orbital evolution `B -> U * B`, the correlation matrix
transforms as

```julia
C -> conj(U) * C * transpose(U)
```

All deterministic formulas must follow this convention. This is the main place where
complex conjugation mistakes are likely, so tests must include at least one complex
mode case.

## Deterministic Equation

`CorrelationLindblad` stores aggregate matrices:

```julia
mutable struct CorrelationLindblad
    h::Hermitian{ComplexF64,Matrix{ComplexF64}}
    damping::Matrix{ComplexF64}          # sum of usual loss/gain Gamma matrices
    source::Matrix{ComplexF64}           # gain source in C convention
    dephasing::Vector{Tuple{Float64,Matrix{ComplexF64}}}  # (gamma, Q) in C convention
    cache::Any                           # implementation-private generator cache
end
```

For a linear mode vector `w = sqrt(gamma) * v`, define the usual single-particle
matrix

```julia
Gamma = w * w'
```

Then:

- loss contributes `Gamma` to `damping`;
- gain contributes `Gamma` to `damping` and `conj(Gamma)` to `source`.

The RHS is:

```julia
dC =
    im * conj(h) * C - im * C * transpose(h)
  + source
  - 0.5 * (conj(damping) * C + C * transpose(damping))
  - 0.5 * sum(gamma * (Q*C + C*Q - 2*Q*C*Q) for (gamma, Q) in dephasing)
```

Here `Q` is a Hermitian projector in the package's `C` convention. For a mode vector
`v`, `Q = conj(v * v')`.

This gives the intended simple limits:

- loss only: vacuum is attractive;
- gain only: full filling is attractive;
- balanced site gain/loss: `C_ss = gamma_gain / (gamma_loss + gamma_gain) * I`;
- pure dephasing: densities are conserved and coherences between measured sectors
  decay.

## Time Evolution

Add:

```julia
lindblad_rhs(L::CorrelationLindblad, C::AbstractMatrix)
lindblad_rhs(L::CorrelationLindblad, s::CorrelationState)

evolve!(s::CorrelationState, L::CorrelationLindblad, dt; method=:expm)
evolve(s::CorrelationState, L::CorrelationLindblad, dt; kwargs...)
```

Initial implementation uses a dense vectorized affine generator:

```julia
d vec(C) / dt = M * vec(C) + b
```

`evolve!` with `method=:expm` forms an augmented matrix exponential so each step is
exact up to dense linear algebra roundoff:

```julia
[vec(C_next); 1] = exp(dt * [M b; 0 0]) * [vec(C); 1]
```

The generated `(M, b)` pair should be cached inside `CorrelationLindblad` and rebuilt
only when the Lindbladian changes. The first implementation may use column-major
`vec`/`reshape` directly, as long as tests cover the convention.

After each step:

- store `s.C = Hermitian(C_next)`;
- symmetrize small roundoff by `C = (C + C')/2` before wrapping;
- do not clamp eigenvalues silently in normal operation;
- optionally expose `check_physical=true` later, but tests should catch unphysical
  output for supported equations.

## Steady State

Add:

```julia
steadystate(L::CorrelationLindblad; atol=1e-10, rtol=1e-8, check_unique=true)
```

It uses the same cached affine generator:

```julia
M * vec(C_ss) + b = 0
```

The direct solver is:

```julia
vec(C_ss) = -M \ b
```

Validation:

- check residual norm `norm(lindblad_rhs(L, C_ss))`;
- wrap as `CorrelationState(Hermitian(C_ss))`;
- check Hermiticity and spectrum within tolerance;
- if the solve is singular or the residual is too large, throw an `ArgumentError`
  explaining that the steady state is not unique or the direct dense solve failed.

This is expected for pure dephasing without loss/gain, where densities are conserved
and the steady state is generally not unique.

## File Integration

Update `src/GaussianFermions.jl` include order:

```julia
LinAlg -> Modes -> States -> Observables -> Hamiltonians -> Channels ->
Trajectory -> CorrelationLindblad -> Quadratic
```

`CorrelationLindblad.jl` depends on `States`, `Hamiltonians`, and `Channels`. It should
not depend on `Quadratic.jl`.

## Tests

Add focused tests to `test/runtests.jl`:

- `lindblad_rhs` gives `im*conj(h)C - im*C*transpose(h)` when there are no baths.
- Deterministic Hamiltonian-only evolution agrees with existing unitary `evolve!`.
- Site loss evolves an initially filled state toward vacuum and `steadystate` is vacuum.
- Site gain evolves vacuum toward full filling and `steadystate` is full filling.
- Balanced gain/loss gives the analytic filling
  `gamma_gain / (gamma_loss + gamma_gain)`.
- Dephasing preserves density and damps off-diagonal correlations.
- `OccupationMonitor` and `HoleMonitor` channel adapters produce the same dephasing RHS.
- A complex mode test verifies the `conj(v * v')` convention.
- A dephasing-only `steadystate` throws a clear non-unique/singular error.

Optionally add one loose comparison between deterministic loss/gain/dephasing evolution
and a trajectory ensemble, but this should not make the test suite slow or flaky.

## Documentation

Update `README.md` with a short usage example:

```julia
H = hopping(16; pbc=true)
L = CorrelationLindblad(H; loss_ops=[sqrt(0.2) * unitvec(16, 1)])
s = CorrelationState(SlaterState(L=16, N=8, config="Z2"))
evolve!(s, L, 1.0)
css = steadystate(L)
```

Use a helper in the example or inline vector construction; do not introduce a public
`unitvec` helper unless it is broadly useful.

## Acceptance Criteria

- `Pkg.test()` passes.
- Existing public APIs continue to work.
- `Quadratic.jl` remains untouched.
- Stage 1b adds deterministic `CorrelationState` Lindblad evolution and steady-state
  solving through both low-level constructors and channel adapters.
- Singular/non-unique steady states fail explicitly rather than returning misleading
  results.
