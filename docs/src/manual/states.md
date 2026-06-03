# States

GaussianFermions.jl has three public state types. The right choice is determined by
which correlations your model can generate.

| State type | Represents | Stored data |
| --- | --- | --- |
| `SlaterState` | Pure number-conserving Gaussian state | occupied orbitals |
| `CorrelationState` | Mixed or pure number-conserving Gaussian state | ``C_{ij}`` |
| `MajoranaState` | General Gaussian state, including pairing | ``\Gamma_{ab}`` |

The package convention for normal correlations is
``C_{ij} = \langle c_i^\dagger c_j\rangle``. The Majorana covariance convention is
given in [Conventions](conventions.md).

## Pure Number-Conserving States

`SlaterState` is the efficient representation for a pure Slater determinant. It stores
an `L x N` matrix of occupied orbitals.

```@example states
using GaussianFermions

s = SlaterState(L=8, N=4, config="Z2")
density(s)
```

The common product-state presets are `"left"`, `"right"`, `"center"`, `"random"`, and
`"Zn"` patterns such as `"Z2"`. You can also construct a state from occupied
positions, a Boolean mask, a Hermitian one-body Hamiltonian at filling `N`, or an
orbital matrix.

```julia
SlaterState([1, 3], 4)
SlaterState([true, false, true, false])
SlaterState(Hermitian(Matrix(hopping(8).h)), 4)
```

`correlation_matrix(s)` forms ``C`` from the orbital matrix. `nmodes(s)` gives `L`,
and `ispure(s)` is true by construction.

## Mixed Number-Conserving States

`CorrelationState` stores the full normal correlation matrix. Use it for mixed
``U(1)``-symmetric states, deterministic Lindblad evolution, and finite-temperature
states.

```@example states
state = CorrelationState(SlaterState(L=6, N=3, config="left"))
particle_number(state)
```

Thermal and maximally mixed states are constructors for common reference states:

```julia
H = hopping(8; pbc=true)
rho = thermalstate(Hermitian(Matrix(H.h)); β=2.0)
infinite_temperature = maximally_mixed(8)
```

If a `CorrelationState` is pure, `SlaterState(state)` converts it back to orbital
form. The conversion throws an `ArgumentError` for a genuinely mixed state.

## Majorana States

`MajoranaState` stores a real antisymmetric covariance matrix and can represent
pairing correlations. It is the state type used with `BdGHamiltonian` and
`MajoranaLindblad`.

```@example states
maj = MajoranaState([1, 0, 1, 0])
normal_correlation(maj)
```

The product-state constructor uses `1` for occupied and `0` for empty sites. A
number-conserving state can be lifted into the Majorana representation:

```@example states
nc = CorrelationState(SlaterState(L=4, N=2, config="Z2"))
maj = MajoranaState(nc)
round.(density(maj); digits=3)
```

Use `normal_correlation(maj)` for ``C`` and `anomalous_correlation(maj)` for ``F``.
For states converted from `SlaterState` or `CorrelationState`, the anomalous
correlation is zero up to numerical tolerance.

## Copying And Accessors

`correlation_matrix` and `covariance_matrix` return matrices that can be inspected
without mutating the state. State-changing operations use bang functions such as
`evolve!`, `measure!`, and `apply_click!`.

See [Free-Fermion Chain](../examples/free-fermion-chain.md) for `SlaterState`
workflows, [BdG Pairing](../examples/bdg-pairing.md) and
[Kitaev Chain](../examples/kitaev-chain.md) for `MajoranaState` workflows, and the
[States & Modes API reference](../reference/states.md) for signatures.
