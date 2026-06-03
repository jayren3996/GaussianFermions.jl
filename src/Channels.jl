#---------------------------------------------------------------------------------------------------
# Dissipation / measurement channels
#
# A channel wraps a Lindblad jump operator built from a quasi-mode `d`, together
# with its rate `γ`. The channel knows three things:
#   jump_rate(ch, s, dt)   -> (probability, work)   work = mode/orbital overlaps
#   apply_click!(ch, s, work)                        realize a quantum jump (mutates a pure state)
#   noclick_operator(ch, dt) / apply_noclick!        the no-jump back-action
# Number-conserving vs number-changing is a *type*, so the (future) deterministic
# fixed-N master equation can accept only `NumberConserving` by dispatch.
#---------------------------------------------------------------------------------------------------
export AbstractChannel, NumberConserving, NumberChanging
export OccupationMonitor, HoleMonitor, Loss, Gain
export dephasing, loss, gain
export jump_rate, apply_click!, noclick_operator, apply_noclick!

abstract type AbstractChannel end
abstract type NumberConserving <: AbstractChannel end
abstract type NumberChanging   <: AbstractChannel end

"""    OccupationMonitor(mode; γ=1) — L = √γ d⁺d (occupation measurement / dephasing)."""
struct OccupationMonitor{T} <: NumberConserving
    mode::QuasiMode{T}
    γ::Float64
end
"""    HoleMonitor(mode; γ=1) — L = √γ dd⁺ (hole-occupation measurement)."""
struct HoleMonitor{T} <: NumberConserving
    mode::QuasiMode{T}
    γ::Float64
end
"""    Loss(mode; γ=1) — L = √γ d (particle loss). Number-changing."""
struct Loss{T} <: NumberChanging
    mode::QuasiMode{T}
    γ::Float64
end
"""    Gain(mode; γ=1) — L = √γ d⁺ (particle gain). Number-changing."""
struct Gain{T} <: NumberChanging
    mode::QuasiMode{T}
    γ::Float64
end

for C in (:OccupationMonitor, :HoleMonitor, :Loss, :Gain)
    @eval $C(m::QuasiMode; γ::Real=1.0) = $C(m, Float64(γ))
end

# single-site convenience constructors
"""    dephasing(i, L; γ=1) — occupation monitor of site `i`."""
dephasing(i::Integer, L::Integer; γ::Real=1.0) = OccupationMonitor(QuasiMode([i], ComplexF64[1], L), Float64(γ))
"""    loss(i, L; γ=1) — particle loss at site `i`."""
loss(i::Integer, L::Integer; γ::Real=1.0) = Loss(QuasiMode([i], ComplexF64[1], L), Float64(γ))
"""    gain(i, L; γ=1) — particle gain at site `i`."""
gain(i::Integer, L::Integer; γ::Real=1.0) = Gain(QuasiMode([i], ComplexF64[1], L), Float64(γ))

#---------------------------------------------------------------------------------------------------
# Jump probabilities  p = γ·dt·⟨L†L⟩  and the work (mode overlaps v = ⟨d|Bₖ⟩)
#
# NOTE the channel form returns the click *probability* `γ dt ⟨L†L⟩` over a step
# `dt` together with a reusable `work` vector, whereas the Majorana
# `jump_rate(s, ℓ)` returns the instantaneous *rate* `⟨L†L⟩` (no `dt`, no work).
#---------------------------------------------------------------------------------------------------
"""
    jump_rate(ch::AbstractChannel, s::SlaterState, dt) -> (probability, work)

Click *probability* `γ dt ⟨L†L⟩` of channel `ch` over a step `dt`, plus the mode
overlaps `work` (pass to [`apply_click!`](@ref)). Contrast the `MajoranaState`
method [`jump_rate(s, ℓ)`](@ref), which returns an instantaneous rate.
"""
# ⟨L†L⟩ = ⟨d†d⟩ = ‖v‖²            for L = d†d (monitor) and L = d (loss)
function jump_rate(ch::Union{OccupationMonitor,Loss}, s::SlaterState, dt::Real)
    v = inner(ch.mode, s.B)
    (real(dot(v, v)) * ch.γ * dt, v)
end
# ⟨L†L⟩ = ⟨dd†⟩ = 1 - ⟨d†d⟩       for L = dd† (hole monitor) and L = d† (gain)
function jump_rate(ch::Union{HoleMonitor,Gain}, s::SlaterState, dt::Real)
    v = inner(ch.mode, s.B)
    ((1 - real(dot(v, v))) * ch.γ * dt, v)
end

#---------------------------------------------------------------------------------------------------
# Quantum-jump (click) actions on a pure state — reuse the orbital-surgery kernels
#---------------------------------------------------------------------------------------------------
function apply_click!(ch::OccupationMonitor, s::SlaterState, v::AbstractVector)  # d†d : occupy d
    replace_vector!(s.B, vector(ch.mode), v); s
end
function apply_click!(ch::HoleMonitor, s::SlaterState, v::AbstractVector)        # dd† : empty d
    avoid_vector!(s.B, vector(ch.mode), v); s
end
function apply_click!(::Loss, s::SlaterState, v::AbstractVector)                 # d : remove a particle
    s.B = delete_vector(s.B, v); s
end
function apply_click!(ch::Gain, s::SlaterState, ::AbstractVector)                # d† : add a particle
    s.B = insert_vector(s.B, vector(ch.mode)); s
end

#---------------------------------------------------------------------------------------------------
# No-jump (no-click) back-action — discrete weak-measurement Kraus operator M₀
#   L†L = d†d  → suppress the occupied component   (particle_exp)
#   L†L = dd†  → suppress the empty component       (hole_exp)
#---------------------------------------------------------------------------------------------------
"""    particle_exp(qm, γdt) — M₀ for L†L = d†d :  √(1-γdt) on d, identity elsewhere."""
function particle_exp(qm::QuasiMode, γdt::Real)
    v = qm.V; n = length(v)
    SMatrix{n,n}((sqrt(1 - γdt) - 1) * v * v' + I(n))
end
"""    hole_exp(qm, γdt) — M₀ for L†L = dd† :  identity on d, √(1-γdt) elsewhere."""
function hole_exp(qm::QuasiMode, γdt::Real)
    v = qm.V; n = length(v); a = sqrt(1 - γdt)
    SMatrix{n,n}((1 - a) * v * v' + a * I(n))
end

noclick_operator(ch::Union{OccupationMonitor,Loss}, dt::Real) = particle_exp(ch.mode, ch.γ * dt)
noclick_operator(ch::Union{HoleMonitor,Gain}, dt::Real)       = hole_exp(ch.mode, ch.γ * dt)

"""
    apply_noclick!(ch, s, dt; normalize=true, threads=true)

Apply the no-jump back-action of `ch` for a step `dt` to a pure state.
"""
function apply_noclick!(ch::AbstractChannel, s::SlaterState, dt::Real; normalize::Bool=true, threads::Bool=true)
    apply!(noclick_operator(ch, dt), s, ch.mode.I; normalize, threads)
    s
end

#---------------------------------------------------------------------------------------------------
# Single-particle L†L block (γ-weighted), for the non-Hermitian effective Hamiltonian
#   d†d ↦ γ |v⟩⟨v|         (occupation monitor, loss)
#   dd† ↦ γ (I - |v⟩⟨v|)   (hole monitor, gain)
#---------------------------------------------------------------------------------------------------
function _proj(ch::AbstractChannel, L::Integer)
    P = zeros(ComplexF64, L, L)
    P[ch.mode.I, ch.mode.I] = ch.mode.V * ch.mode.V'
    P
end
single_particle_LdagL(ch::Union{OccupationMonitor,Loss}, L::Integer) = ch.γ .* _proj(ch, L)
single_particle_LdagL(ch::Union{HoleMonitor,Gain}, L::Integer) =
    ch.γ .* (Matrix{ComplexF64}(I, L, L) .- _proj(ch, L))
