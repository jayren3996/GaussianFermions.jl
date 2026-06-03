#---------------------------------------------------------------------------------------------------
# Quantum-trajectory primitives for Majorana/BdG covariance states
#
# Low-level building blocks for monitored BdG dynamics. There is no trajectory
# "runner": the caller owns the loop (when to evolve, draw a click, measure,
# accumulate) and composes these primitives. The three conditional updates are:
#
#   measure!(s, i)            projective occupation collapse (Schur complement)
#   apply_click!(s, ℓ)        a linear-jump (MCWF) click;  jump_rate(s, ℓ) is its rate
#   apply_noclick!(s, ℓs, dt) the MCWF no-click drift for those jumps
#   weak_measure!(s, i, α)    the weak Gaussian filter exp(α nᵢ) (continuous monitoring)
#
# Each is validated against the number-conserving `SlaterState` and exact
# diagonalization (see test/runtests.jl).
#---------------------------------------------------------------------------------------------------
export weak_measure!, loss_jump, gain_jump, majorana_jump

#---------------------------------------------------------------------------------------------------
# Projective occupation measurement
#
# Measuring mode i selects the 2×2 Majorana block (xᵢ, pᵢ) to the definite-occupation
# covariance σ and conditions the rest by the fermionic-Gaussian Schur complement.
#---------------------------------------------------------------------------------------------------
# Definite-occupation covariance block for the Majorana pair (xᵢ, pᵢ):
# nᵢ = 1 → [0 -1; 1 0] (Γ[i,i+L] = 1-2nᵢ = -1), nᵢ = 0 → [0 1; -1 0].
_occ_block(occupied::Bool) = occupied ? [0.0 -1.0; 1.0 0.0] : [0.0 1.0; -1.0 0.0]

"""
    measure!(s::MajoranaState, i::Integer; rng=Random.default_rng()) -> Bool

Projectively measure the occupation of mode `i` with Born statistics
(`P(occupied) = ⟨nᵢ⟩`), collapse the state, and return the outcome (`true` =
occupied). The covariance is updated by the fermionic-Gaussian measurement rule:
the measured pair becomes a definite-occupation block and the remaining modes are
conditioned by a Schur complement.
"""
function measure!(s::MajoranaState, i::Integer; rng::AbstractRNG=Random.default_rng())
    L = nmodes(s)
    1 ≤ i ≤ L || throw(ArgumentError("mode index $i out of range 1:$L"))
    occupied = rand(rng) < density(s, i)
    _project_occupation!(s, i, occupied)
    occupied
end

function _project_occupation!(s::MajoranaState, i::Integer, occupied::Bool)
    L = nmodes(s)
    Γ = s.Gamma
    A = [i, i + L]
    B = setdiff(1:2L, A)
    σ = _occ_block(occupied)
    ΓAA = Γ[A, A]; ΓAB = Γ[A, B]; ΓBA = Γ[B, A]; ΓBB = Γ[B, B]
    ΓBnew = ΓBB .- ΓBA * ((ΓAA .+ σ) \ ΓAB)            # Schur complement conditioning
    G = zeros(Float64, 2L, 2L)
    G[A, A] .= σ
    G[B, B] .= (ΓBnew .- transpose(ΓBnew)) ./ 2
    s.Gamma = G
    s
end

#---------------------------------------------------------------------------------------------------
# Linear (loss/gain/pairing) quantum jumps — MCWF primitives
#
# A linear Dirac jump L = √γ Σᵢ aᵢcᵢ + bᵢcᵢ⁺ is the Majorana-linear operator L = ℓᵀω,
# with the same complex jump vector ℓ (length 2L) used by `MajoranaLindblad`. On a
# Gaussian state the conditional dynamics stay Gaussian:
#   • G ≡ ⟨ωωᵀ⟩ = I − iΓ is Hermitian; the click rate is N = ℓ†Gℓ (real ≥ 0).
#   • A click sends Γ → Γ + (i/N)(v̄ vᵀ − v v̄ᵀ), v = Gℓ (Wick's theorem; the
#     increment is real antisymmetric — verified analytically and vs ED).
#   • The no-click branch follows the MCWF drift: the deterministic `MajoranaLindblad`
#     dissipator drift for those jumps minus Σ Nₘ(Γ'ₘ − Γ); averaged over the click /
#     no-click branches it reproduces `MajoranaLindblad`.
#---------------------------------------------------------------------------------------------------
"""
    majorana_jump(annihilate, create) -> Vector{ComplexF64}

Majorana jump vector `ℓ` (length `2L`) of the linear Dirac jump
`L = Σᵢ annihilateᵢ cᵢ + createᵢ cᵢ⁺`, in the basis `cᵢ = (xᵢ − i pᵢ)/2`,
`ω = [x₁…x_L, p₁…p_L]`. Pass `ℓ` to [`jump_rate`](@ref) / [`apply_click!`](@ref).
For a rate-`γ` jump fold `√γ` into the amplitudes. Use [`loss_jump`](@ref) /
[`gain_jump`](@ref) for the pure loss/gain cases, this for general pairing baths.
"""
function majorana_jump(annihilate::AbstractVector, create::AbstractVector)
    length(annihilate) == length(create) ||
        throw(ArgumentError("annihilate/create amplitude vectors must have equal length"))
    L = length(annihilate)
    _majorana_jump_vector(ComplexF64.(annihilate), ComplexF64.(create), L)
end

"""    loss_jump(w) -> ℓ — Majorana jump vector of particle loss `L = Σᵢ wᵢ cᵢ`."""
loss_jump(w::AbstractVector) = majorana_jump(w, zeros(ComplexF64, length(w)))
"""    gain_jump(w) -> ℓ — Majorana jump vector of particle gain `L = Σᵢ wᵢ cᵢ⁺`."""
gain_jump(w::AbstractVector) = majorana_jump(zeros(ComplexF64, length(w)), w)

# Click rate N = ℓ†(I − iΓ)ℓ and the working vector v = Gℓ for one jump.
function _jump_rate_and_increment(Γ::AbstractMatrix, ℓ::AbstractVector)
    v = ℓ .- 1im .* (Γ * ℓ)                       # v = (I − iΓ)ℓ = Gℓ
    N = real(dot(ℓ, v))                           # ℓ†Gℓ (real, ≥ 0)
    N, v
end

# Γ' − Γ = (i/N)(v̄ vᵀ − v v̄ᵀ); real antisymmetric.
_jump_increment(N::Real, v::AbstractVector) =
    real.((1im / N) .* (conj(v) * transpose(v) .- v * transpose(conj(v))))

"""
    jump_rate(s::MajoranaState, ℓ::AbstractVector) -> Float64

Instantaneous click rate `⟨L†L⟩ = ℓ†(I − iΓ)ℓ` of the linear jump with Majorana
vector `ℓ` (build `ℓ` with [`loss_jump`](@ref)/[`gain_jump`](@ref)/[`majorana_jump`](@ref)).
The click probability over a step `dt` is `jump_rate(s, ℓ) * dt`.
"""
function jump_rate(s::MajoranaState, ℓ::AbstractVector)
    length(ℓ) == size(s.Gamma, 1) ||
        throw(ArgumentError("jump vector length $(length(ℓ)) does not match 2L = $(size(s.Gamma, 1))"))
    first(_jump_rate_and_increment(s.Gamma, ℓ))
end

"""
    apply_click!(s::MajoranaState, ℓ::AbstractVector) -> s

Apply a definite quantum-jump (click) of the linear jump `ℓ`: render the conditional
Gaussian state `L|ψ⟩/‖L|ψ⟩‖` on the covariance. Errors if the jump rate is zero.
"""
function apply_click!(s::MajoranaState, ℓ::AbstractVector)
    N, v = _jump_rate_and_increment(s.Gamma, ℓ)
    N > 0 || throw(ArgumentError("attempted a zero-rate Majorana jump"))
    G = s.Gamma .+ _jump_increment(N, v)
    s.Gamma = (G .- transpose(G)) ./ 2
    s
end

# MCWF no-click drift: dissipator drift − Σ Nₘ(Γ'ₘ − Γ), summed over linear jumps.
function _noclick_drift(Γ::AbstractMatrix, jumps)
    dΓ = zeros(Float64, size(Γ)...)
    for ℓ in jumps
        N, v = _jump_rate_and_increment(Γ, ℓ)
        B = ℓ * ℓ'
        Br, Bi = real.(B), imag.(B)
        dΓ .+= -2 .* (Br * Γ .+ Γ * Br) .+ 4 .* Bi          # MajoranaLindblad dissipator drift
        N > 0 && (dΓ .-= N .* _jump_increment(N, v))        # remove the average jump contribution
    end
    dΓ
end

"""
    apply_noclick!(s::MajoranaState, jumps, dt) -> s

Apply the MCWF no-click back-action of the linear `jumps` (an iterable of Majorana
jump vectors `ℓ`) over a step `dt`, by RK4-integrating the effective non-Hermitian
drift. Pair with [`jump_rate`](@ref)/[`apply_click!`](@ref) in a trajectory loop:
draw at most one click over `dt` (probability `Σ jump_rate*dt`), otherwise call this.
The click/no-click average reproduces `MajoranaLindblad` built from the same jumps.
"""
function apply_noclick!(s::MajoranaState, jumps, dt::Real)
    Γ = s.Gamma
    k1 = _noclick_drift(Γ, jumps)
    k2 = _noclick_drift(Γ .+ (dt / 2) .* k1, jumps)
    k3 = _noclick_drift(Γ .+ (dt / 2) .* k2, jumps)
    k4 = _noclick_drift(Γ .+ dt .* k3, jumps)
    G = Γ .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    s.Gamma = (G .- transpose(G)) ./ 2
    s
end

#---------------------------------------------------------------------------------------------------
# Continuous (diffusive / quantum-state-diffusion) occupation monitoring
#
# The conditional weak measurement applies the Gaussian filter M = exp(α nᵢ) and
# renormalizes. Writing nᵢ = (1 − Z)/2 with Z = i ωₚω_q (p=i, q=i+L), M ∝ cosh(α/2)I
# − sinh(α/2)Z, and the post-filter 2-point matrix G = ⟨ωωᵀ⟩ = I − iΓ follows from
# Wick's theorem in closed form:
#   G' = (c²G − cs·T₁ + s²·EGE) / (cosh α − w sinh α),   w = Γ[p,q],
#   T₁ = 2wG − i(rₚr_qᵀ − r_qrₚᵀ + r̄ₚr̄_qᵀ − r̄_qr̄ₚᵀ),  rₐ = G[a,:],  E = sign-flip of p,q.
# Exact (no truncation); verified to machine precision vs the number-conserving
# SlaterState filter and vs exact diagonalization (including pairing).
#---------------------------------------------------------------------------------------------------
"""
    weak_measure!(s::MajoranaState, i::Integer, α::Real) -> s

Apply the exact weak-measurement Gaussian filter `exp(α nᵢ)` to the covariance and
renormalize (`α → ±∞` recovers projective occupation/vacancy). For continuous
(quantum-state-diffusion) occupation monitoring at rate `γ`, step it in a loop with
`α = δW + (2⟨nᵢ⟩ − 1) γ dt`, `δW ~ 𝒩(0, γ dt)`; averaging over the noise reproduces
the dephasing Lindblad `D[√γ nᵢ]`.
"""
function weak_measure!(s::MajoranaState, i::Integer, α::Real)
    L = nmodes(s)
    1 ≤ i ≤ L || throw(ArgumentError("mode index $i out of range 1:$L"))
    twoL = 2L
    p = i; q = i + L
    Γ = s.Gamma
    Id = Matrix{ComplexF64}(I, twoL, twoL)
    G = Id .- 1im .* Γ                                       # ⟨ωωᵀ⟩, Hermitian
    w = Γ[p, q]
    rp = G[p, :]; rq = G[q, :]
    X = rp * transpose(rq) .- rq * transpose(rp)
    G1 = (1im / 2) .* (X .+ conj.(X))                        # dG/dα|₀ ; X̄ = c_p c_qᵀ - c_q c_pᵀ
    T1 = 2w .* G .- 2 .* G1
    e = ones(twoL); e[p] = -1.0; e[q] = -1.0
    T2 = (e * transpose(e)) .* G                             # E G E = ε_a ε_b G_ab
    c = cosh(α / 2); sn = sinh(α / 2)
    Gp = (c^2 .* G .- c * sn .* T1 .+ sn^2 .* T2) ./ (cosh(α) - w * sinh(α))
    Gout = real.(1im .* (Gp .- Id))
    s.Gamma = (Gout .- transpose(Gout)) ./ 2
    s
end
