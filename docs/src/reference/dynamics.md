# Dynamics

Time evolution, open-system (Lindblad) solvers, dissipation / measurement channels, and
the low-level quantum-trajectory primitives — covering both the U(1) and Majorana / BdG
layers.

This page collects the deterministic Lindblad generators ([`CorrelationLindblad`](@ref),
[`MajoranaLindblad`](@ref)), the dissipation / measurement channels, and the
quantum-trajectory primitives (projective measurement, quantum-jump / MCWF, and
continuous / QSD weak measurement).

```@autodocs
Modules = [GaussianFermions]
Pages = ["/CorrelationLindblad.jl", "/MajoranaLindblad.jl", "/Channels.jl", "/Trajectory.jl", "/MajoranaTrajectory.jl"]
```
