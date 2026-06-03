# States

GaussianFermions.jl represents fermionic Gaussian states in two complementary ways.

## Number-conserving (U(1)) states

These describe states with a fixed (or fluctuating but ``U(1)``-symmetric) particle
number, fully characterised by the single-particle correlation matrix
``C_{ij} = \langle c_i^\dagger c_j\rangle``.

### `SlaterState`

A pure Slater determinant, stored by its occupied single-particle orbitals.

```julia
s = SlaterState(L=8, N=4, config="Z2")   # half-filled, alternating occupation
nmodes(s)                                # 8
ispure(s)                                # true
```

The `config` keyword selects the initial product configuration (e.g. `"Z2"` for
alternating occupation). You can also build a `SlaterState` directly from a set of
orbitals.

### `CorrelationState`

A general (possibly mixed) number-conserving Gaussian state, stored by its correlation
matrix. Construct it from a `SlaterState`, or from a correlation matrix directly:

```julia
state = CorrelationState(SlaterState(L=8, N=4, config="Z2"))
correlation_matrix(state)                # the L×L matrix Cᵢⱼ
correlation(state, i, j)                 # a single entry
```

Mixed reference states are available as helpers:

```julia
thermalstate(H; β=2.0)                   # Gibbs state of a QuadraticHamiltonian
maximally_mixed(L)                       # infinite-temperature state
```

## General quadratic (BdG / Majorana) states

When pairing terms are present, particle number is no longer conserved and the state is
described by the real antisymmetric Majorana covariance matrix
``\Gamma_{ab} = \tfrac{i}{2}\langle[\omega_a, \omega_b]\rangle``.

### `MajoranaState`

```julia
state = MajoranaState([1, 0, 1, 0])      # product state from occupation pattern
covariance_matrix(state)                 # the 2L×2L covariance Γ
```

From a `MajoranaState` you can recover both normal and anomalous correlations:

```julia
normal_correlation(state)                # Cᵢⱼ = ⟨c⁺ᵢcⱼ⟩
anomalous_correlation(state)             # Fᵢⱼ = ⟨cᵢcⱼ⟩
fermion_correlations(state)              # both, as a Nambu-structured object
```

## Converting between representations

A number-conserving `CorrelationState` embeds into the Majorana representation:

```julia
nc   = CorrelationState(SlaterState(L=4, N=2, config="Z2"))
maj  = MajoranaState(nc)                 # same physical state, covariance form
```

This conversion is the bridge used throughout the package: prepare a state with the
convenient `SlaterState` / `CorrelationState` API, then lift it into `MajoranaState`
when pairing dynamics are required.

See the [States & Spectra API reference](../reference/states.md) for full signatures.
