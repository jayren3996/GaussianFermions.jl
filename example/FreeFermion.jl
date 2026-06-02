include("../src/GaussianFermions.jl")
using LinearAlgebra, Main.GaussianFermions

#--------------------------------------------------------------------------------
# Minimal free-fermion example: unitary evolution + monitored (quantum-jump)
# trajectory on a 1D hopping chain.
#--------------------------------------------------------------------------------
"""
Hopping Hamiltonian on a chain of length `L`:
    H = ∑ᵢ (cᵢ⁺cᵢ₊₁ + h.c.)
"""
function hopping_ham(L::Integer; PBC::Bool=false)
    H = diagm(1 => ones(L-1), -1 => ones(L-1))
    PBC && (H[L, 1] = H[1, L] = 1)
    H
end

"""
Run a single monitored trajectory: alternate unitary hopping with
local occupation measurements (`QJParticle` jumps), returning the
site-resolved density at each step.
"""
function trajectory(; L=32, γ=0.5, dt=0.05, T=20.0, PBC=true)
    U  = evo_operator(Hermitian(hopping_ham(L; PBC)), dt)        # exp(-i H dt)
    js = [QJParticle(QuasiMode([i], [1.0 + 0im], L), γ * dt) for i in 1:L]
    s  = FreeFermionState(L=L, N=L÷2, config="Z2")
    N  = round(Int, T / dt)
    den = zeros(L, N)
    for k in 1:N
        apply!(U, s)            # unitary step
        apply!(js, s)           # one monitoring step (jump or no-jump back-action)
        den[:, k] = diag(s)     # ⟨nᵢ⟩
    end
    den
end

#--------------------------------------------------------------------------------
# Quick sanity check when run as a script.
#--------------------------------------------------------------------------------
let
    s = FreeFermionState(L=8, N=4, config="Z2")
    println("Initial occupation: ", round.(diag(s); digits=3))
    println("Particle number:    ", rank(s))
    println("Half-chain entropy: ", round(ent_S(s, 1:4); digits=4))
    den = trajectory(L=16, T=5.0)
    println("Final occupation (monitored): ", round.(den[:, end]; digits=3))
end
