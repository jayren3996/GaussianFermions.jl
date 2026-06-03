# Observables

All observables of a Gaussian state are functions of its correlation / covariance matrix
and are computed efficiently (polynomial in system size). They work uniformly across
`SlaterState`, `CorrelationState`, and `MajoranaState`.

## Densities and particle number

```julia
state = CorrelationState(SlaterState(L=8, N=4, config="Z2"))

density(state)              # vector of site occupations ⟨nᵢ⟩
density(state, i)           # occupation of a single site
density_profile(state)      # site-resolved density profile
particle_number(state)      # total ⟨N⟩
number_variance(state)      # Var(N)
parity(state)               # fermion parity
purity(state)               # Tr(ρ²)
```

## Correlations

```julia
correlation_profile(state)  # spatial correlation profile
```

For Majorana states, both normal and anomalous correlations are available — see
[Majorana / BdG Foundation](majorana-bdg.md).

## Entanglement

Entanglement of a subregion `A` (given as an index range or vector) is obtained from the
restricted correlation matrix:

```julia
entanglement_entropy(state, 1:4)             # von Neumann entropy of region A
renyi_entropy(state, 1:4; α=2)               # Rényi-α entropy
entanglement_spectrum(state, 1:4)            # single-particle entanglement spectrum
entanglement_hamiltonian_spectrum(state, 1:4)
```

Multipartite information measures:

```julia
mutual_information(state, A, B)              # I(A:B)
tripartite_information(state, A, B, C)       # I₃(A:B:C)
```

These are the standard diagnostics for monitored / measurement-induced transitions; a
trajectory-averaged entanglement entropy is the order parameter in the
[Quantum Trajectories](trajectories.md) examples.

See the [Observables API reference](../reference/observables.md) for full signatures.
