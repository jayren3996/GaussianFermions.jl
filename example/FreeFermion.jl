include("../src/GaussianFermions.jl")
using LinearAlgebra, Main.GaussianFermions

#--------------------------------------------------------------------------------
# Minimal tour of the number-conserving API:
#   states & observables → unitary evolution → a monitored quantum trajectory →
#   an ensemble average.
#--------------------------------------------------------------------------------

# --- states & observables ---
s = SlaterState(L=8, N=4, config="Z2")
println("occupation:        ", round.(density(s); digits=3))
println("particle number:   ", particle_number(s))
println("half-chain S:      ", round(entanglement_entropy(s, 1:4); digits=4))

# --- unitary evolution under a hopping chain ---
H = hopping(8; pbc=true)                 # H = Σ c†ᵢcᵢ₊₁ + h.c.
evolve!(s, H, 1.0)
println("after evolve, S:   ", round(entanglement_entropy(s, 1:4); digits=4),
        "  (N conserved: ", particle_number(s), ")")

# --- a single monitored (quantum-jump) trajectory: occupation monitoring ---
st = SlaterState(L=16, N=8, config="Z2")
chans = [dephasing(i, 16; γ=0.5) for i in 1:16]
for _ in 1:100
    step!(st, hopping(16; pbc=true), chans, 0.05)
end
println("monitored traj S:  ", round(entanglement_entropy(st, 1:8); digits=4))

# --- ensemble average of the entanglement entropy over many trajectories ---
res = ensemble(() -> SlaterState(L=16, N=8, config="Z2"),
               hopping(16; pbc=true),
               [dephasing(i, 16; γ=0.5) for i in 1:16];
               ntraj=64, tspan=5.0, dt=0.05,
               observables=(S = s -> entanglement_entropy(s, 1:8),))
println("ensemble ⟨S⟩(t=5): ", round(res.mean.S[end]; digits=4),
        " ± ", round(res.std.S[end] / sqrt(res.ntraj); digits=4))
