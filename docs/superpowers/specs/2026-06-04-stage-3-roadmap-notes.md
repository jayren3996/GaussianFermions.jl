# Stage 3 Roadmap Notes

## Confirmed Direction

GaussianFermions.jl should stay centered on finite fermionic Gaussian states under
closed, Lindblad, and monitored dynamics. Stage 3A adds lightweight model and
spectral helpers without changing that identity.

## Follow-On Slices

1. General observable engine:
   - `expect_bilinear`
   - `expect_quadratic`
   - `pfaffian_expectation`
   - scoped Wick helpers for Majorana products
   - optional full-counting statistics once the bilinear/Pfaffian API is stable

2. Optional scalability prototype:
   - benchmark current dense `:expm` paths
   - prototype `LinearMaps` plus Krylov action on vectorized covariance state
   - keep Krylov/SciML dependencies weak or extension-only

3. Topology:
   - start with one-dimensional winding/Pfaffian invariants tied to the new model helpers
   - defer full Chern/Z2 machinery or integrate with existing topology packages

4. Trajectory ensemble runner:
   - keep caller-owned loops as the default
   - only add a runner if it is explicitly transparent, optional, and example-backed

## Deferred As Separate Projects

- QuantumLattices-style symbolic model construction
- Kubo/transport response stack
- general Gaussian CP maps
- MPS/tensor-network conversion and compression
