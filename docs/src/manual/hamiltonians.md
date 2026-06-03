# Hamiltonians & Time Evolution

## Quadratic (number-conserving) Hamiltonians

A free-fermion Hamiltonian ``H = \sum_{ij} h_{ij} c_i^\dagger c_j`` is a
`QuadraticHamiltonian`, wrapping the single-particle matrix ``h``. Convenience builders
construct common terms:

```julia
H = hopping(8; pbc=true)                 # nearest-neighbour hopping
μ = chemical_potential(8; μ=0.5)         # on-site chemical potential
```

These return `QuadraticHamiltonian`s whose single-particle matrices can be combined.

## Unitary evolution

`evolve!` advances a state in place under a Hamiltonian for a given time:

```julia
state = CorrelationState(SlaterState(L=8, N=4, config="Z2"))
evolve!(state, H, 0.5)
```

For repeated time steps, precompute a `propagator` (the single-particle evolution
operator ``e^{-iht}``) and reuse it:

```julia
U = propagator(H, 0.05)
for _ in 1:40
    evolve!(state, U)
end
```

`evolve` (without the bang) returns a new evolved state instead of mutating in place.

## BdG (pairing) Hamiltonians

A general quadratic Hamiltonian with pairing,
``H = \sum_{ij} A_{ij} c_i^\dagger c_j + \tfrac12 (B_{ij} c_i^\dagger c_j^\dagger + \text{h.c.})``,
is a `BdGHamiltonian` built from the Hermitian hopping matrix `A` and the antisymmetric
pairing matrix `B`:

```julia
A = zeros(ComplexF64, 4, 4)
B = zeros(ComplexF64, 4, 4); B[1, 2] = 0.3; B[2, 1] = -0.3
H = BdGHamiltonian(A, B)
```

A number-conserving `QuadraticHamiltonian` lifts into a `BdGHamiltonian` directly:

```julia
H = BdGHamiltonian(QuadraticHamiltonian(Matrix(hopping(8; pbc=true).h)))
```

`BdGHamiltonian` evolves a `MajoranaState`'s covariance under dense unitary dynamics, and
supports the same `propagator` / `evolve!` interface:

```julia
state = MajoranaState([1, 0, 1, 0])
evolve!(state, H, 0.5)
```

## Ground and thermal states

Ground and thermal states of a `BdGHamiltonian` come from Bogoliubov / Nambu
diagonalization:

```julia
gs = groundstate(H)                      # BCS ground state as a MajoranaState
ε  = quasiparticle_energies(H)           # Bogoliubov spectrum (≥ 0)
ρβ = thermalstate(H; β=2.0)              # Gibbs state; β → ∞ recovers groundstate(H)
```

See the [Hamiltonians API reference](../reference/hamiltonians.md) and the
[States & Spectra API reference](../reference/states.md).
