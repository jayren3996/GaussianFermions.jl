# Stage 3A Follow-Up Design

## Goal

Make a small follow-up pass after Stage 3A without rewriting the merged work:
update front-door docs, clarify BdG spectrum naming, make zero-mode ground-state
behavior explicit, add optional boundary flux for finite periodic constructors,
and record Wick/Pfaffian observables as the next feature slice.

## Scope

In scope:

- README and docs homepage mention finite model constructors and spectral helpers.
- Docs homepage says the package has three state representations.
- `quasiparticle_spectrum(H)` returns the non-negative BdG quasiparticle
  half-spectrum.
- `nambu_spectrum(H)` returns the full particle-hole-symmetric Nambu spectrum.
- `quasiparticle_energies(H)` remains as a compatibility alias.
- `energy_spectrum(H::BdGHamiltonian)` remains a convenience wrapper over
  `quasiparticle_spectrum`.
- `groundstate(H; zero_mode=...)` exposes an explicit zero-mode policy.
- `ssh_chain`, `aubry_andre_chain`, and `kitaev_chain` accept `flux=0.0` and apply
  it only to periodic boundary bonds.
- Stage 3 roadmap notes put Wick/Pfaffian observables first in the next slice.

Out of scope:

- No topology API.
- No sparse backend.
- No Wick/Pfaffian engine implementation in this pass.
- No breaking removal of `quasiparticle_energies` or `energy_spectrum`.

## API Design

### Spectrum Naming

Add:

```julia
nambu_spectrum(H::BdGHamiltonian) -> Vector{Float64}
quasiparticle_spectrum(H::BdGHamiltonian) -> Vector{Float64}
```

`nambu_spectrum` returns all `2L` Nambu eigenvalues sorted ascending. It is useful
when users need to inspect particle-hole symmetry directly.

`quasiparticle_spectrum` returns the upper half of the sorted Nambu spectrum, so
it has length `L` and preserves exact or near-zero multiplicity without an
artificial clamp.

Existing names stay:

```julia
quasiparticle_energies(H) = quasiparticle_spectrum(H)
energy_spectrum(H::BdGHamiltonian) = quasiparticle_spectrum(H)
```

### Zero-Mode Ground State Policy

Change the BdG ground-state signature to:

```julia
groundstate(H::BdGHamiltonian; zero_mode=:error, atol=1e-10)
groundstate(H::QuadraticHamiltonian; zero_mode=:error, atol=1e-10)
```

Supported policies:

- `:error`: throw `ArgumentError` if any Nambu eigenvalue satisfies
  `abs(epsilon) <= atol`.
- `:empty`: treat zero modes as unoccupied in the Nambu occupation convention.
- `:filled`: treat zero modes as occupied.
- `:half`: half-fill zero modes. This returns a mixed Majorana covariance state,
  matching the thermal zero-temperature limit for exact zero modes.

Defaulting to `:error` makes degenerate parity choices explicit. Existing examples
that use detuned finite systems continue to work.

### Boundary Flux

Add `flux::Real=0.0` to the finite constructors:

```julia
ssh_chain(L; t1=1.0, t2=0.5, pbc=false, flux=0.0)
aubry_andre_chain(L; J=1.0, lambda=1.0, beta=(sqrt(5)-1)/2, phi=0.0,
                  pbc=false, flux=0.0)
kitaev_chain(L; t=1.0, Delta=1.0, mu=0.0, pbc=false, flux=0.0)
```

The actual code keeps the existing Unicode keyword spellings. A nonzero flux with
`pbc=false` throws `ArgumentError`. For periodic systems, the closing bond gets
the phase `exp(im * flux)`. This keeps the finite constructors lightweight and
does not introduce a lattice DSL.

## Documentation

Update:

- `README.md`
- `docs/src/index.md`
- `docs/src/reference/hamiltonians.md`
- `docs/src/manual/hamiltonians.md`
- `docs/src/examples/bdg-pairing.md`
- `docs/src/examples/kitaev-chain.md`
- `docs/superpowers/specs/2026-06-04-stage-3-roadmap-notes.md`

Docs should prefer `quasiparticle_spectrum` for new examples, mention the legacy
alias only in reference/API text, and explain that `groundstate` requires an
explicit zero-mode policy when the BdG spectrum has exact zero modes.

## Testing

Add focused tests for:

- `nambu_spectrum` length/order and particle-hole symmetry.
- `quasiparticle_spectrum` matching the upper half of `nambu_spectrum`.
- `quasiparticle_energies` and `energy_spectrum` compatibility.
- `groundstate` default `:error` on exact BdG zero modes.
- `groundstate(...; zero_mode=:empty)`, `:filled`, and `:half`.
- Invalid `zero_mode` policy.
- Constructor flux on periodic closing bonds.
- Nonzero flux rejected on open chains.

Full verification:

```bash
julia --project=. test/runtests.jl
julia --project=docs docs/make.jl
git diff --check
```

