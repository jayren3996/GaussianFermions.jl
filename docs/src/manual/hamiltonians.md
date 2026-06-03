# Hamiltonians & Time Evolution

Closed-system Gaussian dynamics are represented by single-particle or covariance
propagators. The state representation determines which Hamiltonian type you need.

## Number-Conserving Hamiltonians

`QuadraticHamiltonian` represents

```math
H = \sum_{ij} h_{ij} c_i^\dagger c_j .
```

The matrix `h` is stored as Hermitian. Convenience builders cover the common chain
terms:

```@example hamiltonians
using GaussianFermions, LinearAlgebra

L = 8
H = hopping(L; J=1.0, pbc=true)
mu = chemical_potential(fill(0.2, L))
```

Hamiltonians can be added:

```julia
Htotal = H + mu
```

`evolve!(state, H, t)` applies ``e^{-iht}`` to `SlaterState` or `CorrelationState`.
For fixed-step loops, precompute a propagator:

```@example hamiltonians
state = CorrelationState(SlaterState(L=L, N=4, config="Z2"))
U = propagator(H, 0.05)
for _ in 1:20
    evolve!(state, U)
end
round(particle_number(state); digits=8)
```

`propagator(H, dt)` is cached on `H` for the last `dt`, so repeated fixed-step
evolution avoids recomputing the matrix exponential.

## BdG Hamiltonians

Use `BdGHamiltonian` when pairing terms are present:

```math
H =
\sum_{ij} A_{ij} c_i^\dagger c_j
+ \frac{1}{2}\sum_{ij}
\left(B_{ij} c_i^\dagger c_j^\dagger + \overline{B}_{ij} c_j c_i\right).
```

`A` must be Hermitian and `B` must be antisymmetric.

```@example hamiltonians
A = zeros(ComplexF64, 4, 4)
B = zeros(ComplexF64, 4, 4)
for j in 1:4
    A[j, j] = -0.5                       # chemical potential −μ
end
for j in 1:3
    A[j, j+1] = -1.0; A[j+1, j] = -1.0   # hopping (Hermitian)
    B[j, j+1] =  1.0; B[j+1, j] = -1.0   # p-wave pairing (antisymmetric)
end

bdg = BdGHamiltonian(A, B)
maj = MajoranaState([1, 0, 1, 0])
evolve!(maj, bdg, 0.5)
norm(anomalous_correlation(maj))
```

A number-conserving Hamiltonian can be lifted into BdG form:

```julia
bdg_hopping = BdGHamiltonian(hopping(8; pbc=true))
```

This is useful when a trajectory or Lindblad workflow uses the Majorana
representation but the coherent part remains number-conserving.

## Ground And Thermal States

Ground and thermal states of `BdGHamiltonian` are built by Nambu/Bogoliubov
diagonalization:

```@example hamiltonians
gs  = groundstate(bdg)
eps = quasiparticle_energies(bdg)
rho = thermalstate(bdg; β=2.0)
(min_energy = round(minimum(eps); digits=4),
 gs_number  = round(particle_number(gs); digits=4))
```

`groundstate(bdg)` and `thermalstate(bdg; β)` return `MajoranaState`s. For a
number-conserving Hermitian single-particle Hamiltonian, `thermalstate(h; β, μ=0)`
returns a `CorrelationState`.
