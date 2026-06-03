# Documentation Overhaul Design

## Context

GaussianFermions.jl already has a Documenter.jl site, a `docs/make.jl` build script,
a `docs/Project.toml`, and generated HTML under `docs/build`. The current manual is
thin: most pages are short tours of exported names with a few snippets. That is useful
as a README extension, but it does not yet read like the documentation for a serious
Julia package.

The package itself has a more substantial story than the docs show. It has two
representations of fermionic Gaussian states, number-conserving and Majorana/BdG
Hamiltonians, deterministic Lindblad solvers, and low-level trajectory primitives.
Important convention notes and solver details are currently spread across source
comments, tests, examples, and the README.

The target reader for this overhaul is a physics researcher who already understands
Gaussian fermions and wants to know this package's conventions, modeling choices, and
API contracts. The manual should be conventions-first: equations, index ordering,
sign conventions, and concise explanations, but not long derivations.

## Goals

- Rewrite the documentation so it explains how to model a system with the package,
  not just which functions exist.
- Put all package conventions in one clear place: Dirac correlations, anomalous
  correlations, Majorana basis ordering, covariance sign convention, Hamiltonian
  conventions, Lindblad amplitude conventions, and trajectory update conventions.
- Give researchers enough information to choose between `SlaterState`,
  `CorrelationState`, and `MajoranaState`.
- Give researchers enough information to choose between `CorrelationLindblad`,
  `MajoranaLindblad`, and hand-written trajectory loops.
- Turn the existing examples into documented workflows that users can copy into their
  own scripts.
- Add or improve public docstrings for exported types and functions used by the
  manual and reference pages.
- Replace broad source-order `@autodocs` pages with intentional, grouped reference
  pages.
- Keep the documentation build reliable in CI.

## Non-Goals

- No package API redesign as part of the documentation overhaul.
- No change to physical conventions unless the documentation work exposes a concrete
  bug that must be fixed separately.
- No long textbook derivations. The manual can state the equations used by the
  package, but it should not become a theory review.
- No hidden trajectory ensemble runner. The docs should be explicit that callers own
  trajectory loops.
- No attempt to document internal helpers that are not part of the public API.

## Documentation Architecture

Keep Documenter.jl as the documentation system. Reorganize the site around four reader
paths: orientation, manual, examples, and API reference.

### Orientation

The home page should answer four questions quickly:

- What does GaussianFermions.jl simulate?
- Which representation should I use?
- Which dynamics are available?
- What is deliberately left to the caller?

The home page should include a small decision table:

```text
Need fixed-particle pure states           -> SlaterState
Need mixed U(1)-symmetric states          -> CorrelationState
Need pairing / BdG / anomalous correlators -> MajoranaState
Need deterministic U(1) open dynamics     -> CorrelationLindblad
Need deterministic BdG open dynamics      -> MajoranaLindblad
Need monitored trajectories               -> write a loop from primitives
```

The page should link to the manual for conventions, examples for workflows, and the
API reference for signatures.

### Manual

The manual should be the main reader path for physics researchers. It should use
equations and small runnable snippets, but every page should explain the modeling
choice before showing code.

Proposed pages:

- `manual/conventions.md`
- `manual/states.md`
- `manual/hamiltonians.md`
- `manual/lindblad.md`
- `manual/trajectories.md`
- `manual/observables.md`

`manual/conventions.md` should be new and should carry the sign/index burden for the
whole site. It should define:

- Site indices are Julia 1-based indices.
- Dirac operators use the package convention
  ``C_{ij} = \langle c_i^\dagger c_j\rangle``.
- Anomalous correlations use
  ``F_{ij} = \langle c_i c_j\rangle``.
- The Majorana basis is ordered as
  ``\omega = [x_1,\ldots,x_L,p_1,\ldots,p_L]``.
- ``x_j = c_j + c_j^\dagger`` and
  ``p_j = i(c_j - c_j^\dagger)``.
- The covariance is
  ``\Gamma_{ab} = \frac{i}{2}\langle[\omega_a,\omega_b]\rangle``.
- Lindblad loss/gain vectors are amplitude vectors: for rate ``\gamma`` on a
  normalized mode ``v``, pass ``\sqrt{\gamma}v``.
- `dephasing_ops` entries are mode/rate pairs where the constructor accepts that form.
- Mutating functions use Julia's bang convention, for example `evolve!` and
  `apply_click!`.

`manual/states.md` should explain the state hierarchy:

- `SlaterState`: pure number-conserving state stored by occupied orbitals.
- `CorrelationState`: possibly mixed number-conserving state stored by ``C``.
- `MajoranaState`: general Gaussian state stored by ``\Gamma``.
- Conversion from number-conserving states to Majorana covariance form.
- Purity, `nmodes`, and when `SlaterState(correlation_state)` is valid.

`manual/hamiltonians.md` should explain:

- `QuadraticHamiltonian` for ``\sum h_{ij} c_i^\dagger c_j``.
- Convenience builders such as `hopping` and `chemical_potential`.
- `propagator` caching and when it is useful.
- `BdGHamiltonian` for hopping plus pairing blocks.
- `groundstate`, `thermalstate`, and `quasiparticle_energies`.

`manual/lindblad.md` should replace the current split between correlation Lindblad
and part of the Majorana/BdG page. It should explain:

- What closes at the level of ``C`` for `CorrelationLindblad`.
- What closes at the level of ``\Gamma`` for `MajoranaLindblad`.
- Loss, gain, dephasing, and pairing bath inputs.
- The relation between deterministic Lindblad evolution and stochastic
  unravelings.
- `steadystate` for finite systems with an attractive steady state.
- Dense finite-system scope.

`manual/trajectories.md` should keep the current low-level emphasis, but make the
contract sharper:

- There is no ensemble runner.
- A trajectory step is composed from evolution, rates, click/no-click updates, weak
  measurements, and caller-owned accumulation.
- Number-conserving channels use `OccupationMonitor`, `HoleMonitor`, `Loss`, `Gain`,
  and channel forms of `jump_rate`, `apply_click!`, `apply_noclick!`.
- Majorana/BdG linear jumps use `loss_jump`, `gain_jump`, `majorana_jump`, and the
  Majorana forms of the same primitives.
- Stochastic examples should use fixed RNG seeds when shown as executable examples.

`manual/observables.md` should explain what is computed from ``C`` or ``\Gamma``:

- Densities and particle number.
- Normal and anomalous correlations.
- Number variance, parity, and purity.
- Entanglement entropy, Renyi entropy, entanglement spectrum, and entanglement
  Hamiltonian spectrum.
- Mutual and tripartite information for monitored-dynamics diagnostics.

### Examples

Add an examples/workflows section under `docs/src/examples`. These pages should be
more narrative than the API reference and should be based on existing scripts and
tests.

Proposed pages:

- `examples/free-fermion-chain.md`
- `examples/deterministic-lindblad.md`
- `examples/trajectories-vs-lindblad.md`
- `examples/bdg-pairing.md`
- `examples/monitored-mutual-information.md`

The examples should be small enough to run during documentation builds unless they
are intentionally stochastic or expensive. Longer stochastic examples should be
presented as regular code blocks with fixed seeds and a note explaining which example
file contains the full run.

### API Reference

Replace broad `@autodocs` reference pages with grouped `@docs` blocks. The grouping
should be intentional and reader-facing:

- States and conversions.
- Hamiltonians and propagators.
- Lindblad solvers and channels.
- Trajectory primitives.
- Observables.
- Low-level linear algebra helpers only if they are genuinely public.

Reference pages should not duplicate the manual. They should state signatures,
arguments, mutation behavior, return values, conventions, and a short example where
that prevents ambiguity.

## Docstring Scope

Add or improve docstrings for exported public API. At minimum, cover:

- State types and constructors:
  `SlaterState`, `CorrelationState`, `MajoranaState`, `QuasiMode`.
- State accessors and conversions:
  `correlation_matrix`, `correlation`, `covariance_matrix`,
  `fermion_correlations`, `normal_correlation`, `anomalous_correlation`, `nmodes`,
  `ispure`, `thermalstate`, `maximally_mixed`.
- Hamiltonians:
  `QuadraticHamiltonian`, `BdGHamiltonian`, `hopping`, `chemical_potential`,
  `propagator`, `groundstate`, `quasiparticle_energies`.
- Evolution:
  `evolve`, `evolve!`, `Gate`, `apply!`.
- Observables:
  `density`, `density_profile`, `particle_number`, `number_variance`, `purity`,
  `parity`, `correlation_profile`, `entanglement_entropy`, `renyi_entropy`,
  `mutual_information`, `tripartite_information`, `entanglement_spectrum`,
  `entanglement_hamiltonian_spectrum`.
- Lindblad and channels:
  `CorrelationLindblad`, `MajoranaLindblad`, `lindblad_rhs`,
  `majorana_lindblad_rhs`, `steadystate`, `OccupationMonitor`, `HoleMonitor`,
  `Loss`, `Gain`, `dephasing`, `loss`, `gain`.
- Trajectory primitives:
  `measure!`, `weak_measure!`, `Feedback`, `jump_rate`, `apply_click!`,
  `noclick_operator`, `apply_noclick!`, `NonHermitianGenerator`,
  `effective_hamiltonian`, `noclick_propagator`, `evolve_noclick!`, `loss_jump`,
  `gain_jump`, `majorana_jump`.

Docstrings should be concise. The preferred shape is:

````julia
"""
    function_name(args...; kwargs...)

One or two sentences explaining what the function does and which convention it uses.

Mention whether the function mutates its input, what it returns, and any important
rate/sign/index convention.

# Example

```julia
...
```
"""
````

Long derivations belong in neither docstrings nor manual pages for this pass.

## Validation and CI

The documentation build should remain a normal Documenter.jl build:

```bash
julia --project=docs docs/make.jl
```

Stable snippets should use `@example` blocks where possible. Stochastic snippets should
either set a fixed RNG seed or remain plain fenced `julia` blocks if exact output is
not worth checking.

The docs CI should fail on build errors. A strict `checkdocs = :exports` policy is a
good end state, but it can be introduced only after the docstring pass covers the
public API. Until then, the build should avoid false failures while still catching
broken pages, broken examples, and syntax errors.

## Implementation Boundaries

The overhaul should preserve the existing deployment direction unless the user asks
to change it. The current workspace already has uncommitted changes in
`.github/workflows/documentation.yml` and `docs/make.jl`; implementation should treat
those as pre-existing user/workspace changes and avoid overwriting them accidentally.

No generated `docs/build` files should be committed unless the repository already
expects generated docs to be tracked for a specific reason.

## Acceptance Criteria

- The docs site has a conventions-first manual suitable for physics researchers.
- The home page gives a clear representation/dynamics decision guide.
- Manual pages explain package conventions before code examples.
- Existing examples are represented as documented workflows.
- API reference pages use grouped `@docs` blocks rather than source-order dumps.
- Exported public API used by the manual has useful docstrings.
- Documentation examples that are checked by Documenter are deterministic.
- `julia --project=docs docs/make.jl` builds successfully.
- The implementation does not overwrite unrelated uncommitted changes.
