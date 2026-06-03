# Conventions

This page collects the conventions used throughout GaussianFermions.jl. The package
does not try to infer conventions from input names; the matrices and vectors you pass
are interpreted according to the definitions below.

## Indices

Site and mode indices are Julia indices, so they start at `1`. A region such as
`1:4` means the first four sites in the current single-particle basis.

## Dirac Correlations

For number-conserving Gaussian states the stored object is the normal correlation
matrix

```math
C_{ij} = \langle c_i^\dagger c_j\rangle .
```

`correlation_matrix(s)` returns this matrix for `SlaterState` and
`CorrelationState`. For a pure Slater determinant with orbital matrix `B`, the package
uses

```math
C = \overline{B} B^T .
```

The anomalous correlation used by the Majorana layer is

```math
F_{ij} = \langle c_i c_j\rangle .
```

For number-conserving states, `F` is zero up to numerical tolerance after conversion
to `MajoranaState`.

## Majorana Basis

The Majorana basis is ordered as

```math
\omega = [x_1,\ldots,x_L,p_1,\ldots,p_L],
```

with

```math
x_j = c_j + c_j^\dagger,\qquad
p_j = i(c_j - c_j^\dagger).
```

The covariance matrix is

```math
\Gamma_{ab} = \frac{i}{2}\langle[\omega_a,\omega_b]\rangle .
```

`covariance_matrix(s)` returns a copy of the stored covariance. `normal_correlation`
and `anomalous_correlation` convert the covariance back to the package's `C` and `F`
conventions.

## Hamiltonians

`QuadraticHamiltonian` represents a number-conserving Hamiltonian

```math
H = \sum_{ij} h_{ij} c_i^\dagger c_j .
```

The single-particle propagator is

```math
U(t) = e^{-i h t}.
```

`BdGHamiltonian` represents the general quadratic Hamiltonian

```math
H =
\sum_{ij} A_{ij} c_i^\dagger c_j
+ \frac{1}{2}\sum_{ij}
\left(B_{ij} c_i^\dagger c_j^\dagger + \overline{B}_{ij} c_j c_i\right),
```

where `A` is Hermitian and `B` is antisymmetric.

## Lindblad Inputs

Loss and gain inputs are amplitude vectors. For rate ``\gamma`` acting on a normalized
mode ``v``, pass

```math
\sqrt{\gamma}\,v .
```

For example, loss at site `1` with rate `0.2` on a four-site system is

```julia
loss1 = sqrt(0.2) * ComplexF64[1, 0, 0, 0]
```

Where `dephasing_ops` accepts mode/rate pairs, each entry has the form `(v, gamma)`.
The mode `v` should be normalized.

## Mutation

Functions ending in `!` mutate their state argument. The common examples are
`evolve!`, `apply!`, `measure!`, `weak_measure!`, `apply_click!`, and
`apply_noclick!`. The non-bang `evolve` returns an evolved copy.
