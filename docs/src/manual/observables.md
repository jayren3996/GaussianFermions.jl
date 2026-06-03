# Observables

Gaussian observables are computed from the restricted normal correlation matrix
``C_A`` or the restricted Majorana covariance matrix ``\Gamma_A``. The same public
functions work for `SlaterState`, `CorrelationState`, and `MajoranaState` where the
quantity is defined.

## Density And Number Diagnostics

```@example observables
using GaussianFermions

s = CorrelationState(SlaterState(L=8, N=4, config="Z2"))
density(s)
```

Useful one-body diagnostics include:

```julia
density(s)              # vector of occupations
density(s, i)           # one site
density_profile(s)      # alias for density(s)
particle_number(s)      # total expectation value
number_variance(s)      # Var(N)
parity(s)               # fermion parity expectation
purity(s)               # Tr(rho^2)
```

For `MajoranaState`, `density` is computed from the Majorana covariance using the
same ``C_{ii}`` convention as the number-conserving layer.

## Correlations

For number-conserving states, use `correlation_matrix` or `correlation`.

```julia
C = correlation_matrix(s)
c13 = correlation(s, 1, 3)
```

For Majorana states, use:

```julia
C = normal_correlation(maj)
F = anomalous_correlation(maj)
```

`correlation_profile(s, i)` returns correlations from a chosen site to all sites.

## Entanglement And Spectra

Entanglement functions take a region as a range or vector of site indices:

```@example observables
entanglement_entropy(s, 1:4)
```

The package provides:

```julia
entanglement_entropy(s, A)
renyi_entropy(s, A; α=2)
entanglement_spectrum(s, A)
entanglement_hamiltonian_spectrum(s, A)
```

For a `SlaterState`, the von Neumann entropy has a pure-state orbital fast path. Mixed
number-conserving states and Majorana states go through restricted correlation or
covariance spectra.

## Information Measures

Multipartite diagnostics are central to monitored dynamics. They are nonlinear in the
state, so a product state has none — correlations have to be built up first:

```@example observables
evolved = CorrelationState(SlaterState(L=8, N=4, config="Z2"))
evolve!(evolved, hopping(8; pbc=true), 1.0)

A = 1:2; B = 3:4; C = 5:6
(mutual     = round(mutual_information(evolved, A, B); digits=4),
 tripartite = round(tripartite_information(evolved, A, B, C); digits=4))
```

!!! note
    Because `mutual_information` and `tripartite_information` are nonlinear, in
    trajectory studies the caller computes them inside the sampling loop and averages
    over trajectories or a late-time window. `tripartite_information` is the
    scale-invariant order parameter used in the
    [Measurement-Induced Transition](../examples/measurement-induced-transition.md)
    and [Mutual & Tripartite Information](../examples/monitored-mutual-information.md)
    examples.
