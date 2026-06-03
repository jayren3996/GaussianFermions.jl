#---------------------------------------------------------------------------------------------------
# Evolution, gates, and the quantum-trajectory engine
#---------------------------------------------------------------------------------------------------
export evolve, evolve!, Gate, apply!
export measure!, weak_measure!, Feedback
export NonHermitianGenerator, effective_hamiltonian, noclick_propagator, evolve_noclick!

#---------------------------------------------------------------------------------------------------
# Re-orthonormalization (pure states only)
#---------------------------------------------------------------------------------------------------
"""    normalize!(s::SlaterState) — re-orthonormalize the orbitals via QR."""
function LinearAlgebra.normalize!(s::SlaterState)
    s.B = Matrix(qr!(s.B).Q)
    s
end

#---------------------------------------------------------------------------------------------------
# Whole-system unitary evolution
#---------------------------------------------------------------------------------------------------
"""
    evolve!(s, U)            apply a single-particle unitary `U`
    evolve!(s, H, dt)        evolve for time `dt` under a `QuadraticHamiltonian`
    evolve(s, ...)           non-mutating copy

Pure (`SlaterState`): `B ← U B`. Mixed (`CorrelationState`): `C ← U C U'`.
"""
function evolve!(s::SlaterState, U::AbstractMatrix)
    s.B = U * s.B
    s
end
function evolve!(s::CorrelationState, U::AbstractMatrix)
    # orbitals evolve B → U B, so C = conj(B)Bᵀ transforms as C → Ū C Uᵀ
    s.C = Hermitian(conj(U) * s.C * transpose(U))
    s
end
evolve!(s::NumberConservingGaussianState, H::QuadraticHamiltonian, dt::Real) = evolve!(s, propagator(H, dt))
evolve(s::NumberConservingGaussianState, args...) = evolve!(copy(s), args...)

#---------------------------------------------------------------------------------------------------
# Local gates
#---------------------------------------------------------------------------------------------------
"""    Gate(M, I) — a local single-particle gate `M` acting on sites `I`."""
struct Gate{T<:AbstractMatrix}
    M::T
    I::Vector{Int64}
end

"""
    apply!(M, s::SlaterState, inds; normalize=false, threads=false)

Apply a local single-particle matrix `M` on sites `inds` (in place, turbo kernels).
"""
function apply!(M::AbstractMatrix, s::SlaterState, inds::AbstractVector{<:Integer};
               normalize::Bool=false, threads::Bool=false)
    B = s.B[inds, :]
    Bv = view(s.B, inds, :)
    threads ? tturbo_mul!(Bv, M, B) : turbo_mul!(Bv, M, B)
    normalize && normalize!(s)
    s
end
function apply!(M::AbstractMatrix, s::CorrelationState, inds::AbstractVector{<:Integer}; kwargs...)
    # consistent with B → M B on the support: C → conj(M) C transpose(M) on `inds`
    C = Matrix(s.C)
    C[inds, :] = conj(M) * C[inds, :]
    C[:, inds] = C[:, inds] * transpose(M)
    s.C = Hermitian(C)
    s
end
apply!(g::Gate, s::NumberConservingGaussianState; kwargs...) = apply!(g.M, s, g.I; kwargs...)

function apply!(gates::AbstractVector{<:Gate}, s::NumberConservingGaussianState;
               normalize::Bool=false, threads::Bool=length(gates) > 100)
    if threads
        Threads.@threads for g in gates
            apply!(g, s)
        end
    else
        for g in gates
            apply!(g, s)
        end
    end
    normalize && normalize!(s)
    s
end

#---------------------------------------------------------------------------------------------------
# Quantum-jump (MCWF) trajectories are composed in user scripts from the channel
# primitives in Channels.jl — `jump_rate(ch, s, dt)`, `apply_click!(ch, s, work)`,
# `apply_noclick!(ch, s, dt)` — together with `evolve!`. A minimal loop:
#
#     evolve!(s, H, dt)
#     for ch in channels
#         rate, work = jump_rate(ch, s, dt)
#         rand(rng) < rate ? apply_click!(ch, s, work) : apply_noclick!(ch, s, dt)
#         normalize!(s)
#     end
#
# See example/FreeFermion.jl and example/Dephase.jl for full trajectory + averaging loops.
#---------------------------------------------------------------------------------------------------
# No-click / non-Hermitian deterministic evolution (post-selected trajectory)
#---------------------------------------------------------------------------------------------------
"""    NonHermitianGenerator(Heff) — wraps the effective single-particle Hamiltonian."""
struct NonHermitianGenerator{T}
    Heff::Matrix{T}
end

"""
    effective_hamiltonian(H::QuadraticHamiltonian, channels) -> NonHermitianGenerator

`H_eff = h - (i/2) Σ L†L` (single-particle), the generator of the no-click branch.
"""
function effective_hamiltonian(H::QuadraticHamiltonian, channels)
    L = nmodes(H)
    Heff = Matrix{ComplexF64}(Matrix(H.h))
    for ch in channels
        Heff .-= (im / 2) .* single_particle_LdagL(ch, L)
    end
    NonHermitianGenerator(Heff)
end

"""    noclick_propagator(G, dt) = exp(-i H_eff dt)  (non-unitary)."""
noclick_propagator(G::NonHermitianGenerator, dt::Real) = exp(-im * dt * G.Heff)

"""
    evolve_noclick!(s::SlaterState, G, dt; renormalize=true) -> w

Deterministic no-click evolution `B ← exp(-i H_eff dt) B`, returning the no-click
weight `w = ‖unnormalized‖²` of the step (accumulate `log(w)` along a trajectory).
"""
function evolve_noclick!(s::SlaterState, G::NonHermitianGenerator, dt::Real; renormalize::Bool=true)
    Bnew = noclick_propagator(G, dt) * s.B
    w = real(det(Bnew' * Bnew))
    s.B = Bnew
    renormalize && normalize!(s)
    w
end

#---------------------------------------------------------------------------------------------------
# Continuous (diffusive / quantum-state-diffusion) monitoring
#---------------------------------------------------------------------------------------------------
"""
    weak_measure!(s::SlaterState, i::Integer, α::Real) -> s

Apply the weak-measurement Gaussian filter `exp(α nᵢ)` to site `i` and renormalize
(`B[i,:] ← e^α B[i,:]`). For continuous (quantum-state-diffusion) occupation
monitoring at rate `γ`, step it in a loop with `α = δW + (2⟨nᵢ⟩ − 1) γ dt`,
`δW ~ 𝒩(0, γ dt)`; averaging over the noise reproduces the dephasing Lindblad
`D[√γ nᵢ]`. Matches the `MajoranaState` [`weak_measure!`](@ref) to machine precision.
"""
function weak_measure!(s::SlaterState, i::Integer, α::Real)
    apply!(fill(ComplexF64(exp(α)), 1, 1), s, [i]; normalize=true)
    s
end

"""
    weak_measure!(s::SlaterState, qm::QuasiMode, α::Real) -> s

Weak-measurement filter `exp(α nₘ)` for the occupation `nₘ = d†d` of quasi-mode
`qm` (`d = Σₐ qm.Vₐ c_{qm.Iₐ}`); `B[qm.I,:] ← [(e^α-1) V V' + I] B[qm.I,:]`,
renormalized. Read `⟨nₘ⟩` with [`density(s, qm)`](@ref) to form `α` in a QSD loop.
"""
function weak_measure!(s::SlaterState, qm::QuasiMode, α::Real)
    apply!((exp(α) - 1) .* qm.V * qm.V' + I, s, qm.I; normalize=true)
    s
end

#---------------------------------------------------------------------------------------------------
# Projective measurement
#---------------------------------------------------------------------------------------------------
"""
    measure!(qm::QuasiMode, s::SlaterState; rng) -> Bool

Projective occupation measurement of mode `qm` with Born statistics; returns the
outcome (`true` = occupied).
"""
function measure!(qm::QuasiMode, s::SlaterState; rng::AbstractRNG=Random.default_rng())
    p = inner(qm, s.B)
    if rand(rng) < real(dot(p, p))
        replace_vector!(s.B, vector(qm), p); normalize!(s); return true
    else
        avoid_vector!(s.B, vector(qm), p); normalize!(s); return false
    end
end

#---------------------------------------------------------------------------------------------------
# Measurement-conditioned feedback
#---------------------------------------------------------------------------------------------------
"""
    Feedback(channel, on_click, on_noclick)

A channel whose click / no-click outcome triggers a feedback unitary
(`on_click` / `on_noclick`) on the channel's support.
"""
struct Feedback{C<:AbstractChannel,U<:AbstractMatrix}
    channel::C
    on_click::U
    on_noclick::U
end

"""    apply!(fb::Feedback, s::SlaterState, dt; rng) -> Bool (whether a click occurred)."""
function apply!(fb::Feedback, s::SlaterState, dt::Real; rng::AbstractRNG=Random.default_rng())
    ch = fb.channel
    rate, v = jump_rate(ch, s, dt)
    if rand(rng) < rate
        apply_click!(ch, s, v); normalize!(s)
        apply!(fb.on_click, s, ch.mode.I; threads=true)
        return true
    else
        apply_noclick!(ch, s, dt; normalize=true)
        apply!(fb.on_noclick, s, ch.mode.I; threads=true)
        return false
    end
end

#---------------------------------------------------------------------------------------------------
# Trajectory averaging is the caller's responsibility — there is no ensemble runner.
# Loop over trajectories in your script, stepping with the primitives above and
# accumulating whatever observables you need. See example/FreeFermion.jl (entanglement
# entropy) and example/MI.jl (mutual information) for averaging loops.
#---------------------------------------------------------------------------------------------------
