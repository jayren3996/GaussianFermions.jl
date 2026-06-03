#---------------------------------------------------------------------------------------------------
# Quantum trajectories for Majorana/BdG covariance states
#
# Building blocks for monitored BdG dynamics. Projective occupation measurement is the
# fundamental conditional update: measuring mode i selects the 2×2 Majorana block
# (xᵢ, pᵢ) to the definite-occupation covariance σ and conditions the rest by the
# fermionic-Gaussian Schur complement. Verified against the number-conserving
# `SlaterState` measurement.
#---------------------------------------------------------------------------------------------------
export measure!, step!, step_diffusive!, MajoranaJumps

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

"""
    step!(s::MajoranaState, O::AbstractMatrix, monitors, dt; rng=Random.default_rng())
    step!(s::MajoranaState, H::BdGHamiltonian, monitors, dt; rng=Random.default_rng())

One monitored-trajectory step: apply the unitary (covariance propagator `O`, or
`propagator(H, dt)`), then projectively measure each monitored site. `monitors` is an
iterable of `(site, rate)` pairs; site `i` is measured with probability `rate*dt`.

Projective measurement of `nᵢ` at rate `γ` averages to the dephasing Lindblad
`D[√(2γ) nᵢ]`, so the trajectory ensemble reproduces
`MajoranaLindblad(H; dephasing_ops=[(eᵢ, 2γ)])` (verified against `MajoranaLindblad`).
"""
function step!(s::MajoranaState, O::AbstractMatrix, monitors, dt::Real; rng::AbstractRNG=Random.default_rng())
    evolve!(s, O)
    for (i, γ) in monitors
        rand(rng) < γ * dt && measure!(s, i; rng)
    end
    s
end

step!(s::MajoranaState, H::BdGHamiltonian, monitors, dt::Real; rng::AbstractRNG=Random.default_rng()) =
    step!(s, propagator(H, dt), monitors, dt; rng)

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
# Linear (loss/gain/pairing) quantum-jump unraveling (MCWF) on the covariance
#
# A linear Dirac jump L = √γ Σᵢ aᵢcᵢ + bᵢcᵢ⁺ is the Majorana-linear operator
# L = ℓᵀω with the same jump vector ℓ used by `MajoranaLindblad`. On a Gaussian
# state the conditional dynamics stay Gaussian:
#   • G ≡ ⟨ωωᵀ⟩ = I − iΓ is Hermitian; the click rate is N = ℓ†Gℓ (real ≥ 0).
#   • A click sends Γ → Γ + (i/N)(v̄ vᵀ − v v̄ᵀ) with v = Gℓ  (Wick's theorem;
#     the increment is real antisymmetric — verified analytically and vs ED).
#   • The no-click branch follows the MCWF drift: the deterministic `MajoranaLindblad`
#     dissipator drift for these jumps minus Σ N_m(Γ'_m − Γ). By construction the
#     trajectory average reproduces `MajoranaLindblad` (verified).
#---------------------------------------------------------------------------------------------------
"""
    MajoranaJumps(L; loss_ops=[], gain_ops=[], majorana_ops=[], monitors=[])

Quantum-jump (MCWF) channel set for monitored `MajoranaState` trajectories on an
`L`-mode system. `loss_ops`/`gain_ops`/`majorana_ops` are the linear jump operators
(same convention as [`MajoranaLindblad`](@ref)): a click renders the conditional
Gaussian state, the no-click branch follows the effective non-Hermitian drift.
`monitors` is a list of `(site, rate)` projective occupation measurements (as in the
projective [`step!`](@ref)). Plugs into the existing [`ensemble`](@ref) runner;
the trajectory average reproduces the matching `MajoranaLindblad`.
"""
struct MajoranaJumps
    jumps::Vector{Vector{ComplexF64}}        # linear Majorana jump vectors ℓ (length 2L)
    monitors::Vector{Tuple{Int,Float64}}     # projective occupation monitors (site, rate)
    twoL::Int
end

function MajoranaJumps(L::Integer; loss_ops=[], gain_ops=[], majorana_ops=[], monitors=[])
    twoL = 2L
    jumps = Vector{ComplexF64}[]
    for w in loss_ops
        v = _dense_mode_vector(w, L)
        push!(jumps, _majorana_jump_vector(v, zeros(ComplexF64, L), L))
    end
    for w in gain_ops
        v = _dense_mode_vector(w, L)
        push!(jumps, _majorana_jump_vector(zeros(ComplexF64, L), v, L))
    end
    for l in majorana_ops
        length(l) == twoL ||
            throw(ArgumentError("Majorana jump vector length $(length(l)) does not match 2L = $twoL"))
        lc = Vector{ComplexF64}(l)
        all(isfinite, lc) || throw(ArgumentError("Majorana jump vector entries must be finite"))
        push!(jumps, lc)
    end
    mon = Tuple{Int,Float64}[]
    for m in monitors
        length(m) == 2 || throw(ArgumentError("monitors must be (site, rate) pairs"))
        i, γ = m
        (1 ≤ Int(i) ≤ L) || throw(ArgumentError("monitor site $i out of range 1:$L"))
        γf = Float64(γ); (isfinite(γf) && γf ≥ 0) ||
            throw(ArgumentError("monitor rate must be finite and non-negative, got $γf"))
        push!(mon, (Int(i), γf))
    end
    MajoranaJumps(jumps, mon, twoL)
end

# Click rate N = ℓ†(I − iΓ)ℓ and the no-click target increment Γ' − Γ for one jump.
function _jump_rate_and_increment(Γ::AbstractMatrix, ℓ::AbstractVector)
    v = ℓ .- 1im .* (Γ * ℓ)                       # v = (I − iΓ)ℓ = Gℓ
    N = real(dot(ℓ, v))                           # ℓ†Gℓ (real, ≥ 0)
    N, v
end

function _jump_increment(N::Real, v::AbstractVector)
    # Γ' − Γ = (i/N)(v̄ vᵀ − v v̄ᵀ); real antisymmetric.
    inc = (1im / N) .* (conj(v) * transpose(v) .- v * transpose(conj(v)))
    real.(inc)
end

# Apply a definite click of jump `ℓ` to the conditional covariance.
function _apply_majorana_jump!(s::MajoranaState, ℓ::AbstractVector)
    N, v = _jump_rate_and_increment(s.Gamma, ℓ)
    N > 0 || throw(ArgumentError("attempted a zero-rate Majorana jump"))
    G = s.Gamma .+ _jump_increment(N, v)
    s.Gamma = (G .- transpose(G)) ./ 2
    s
end

# MCWF no-click drift: dissipator drift − Σ N_m (Γ'_m − Γ), summed over linear jumps.
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

# Integrate the no-click drift over `dt` (RK4) and re-antisymmetrize.
function _noclick_step!(s::MajoranaState, jumps, dt::Real)
    Γ = s.Gamma
    k1 = _noclick_drift(Γ, jumps)
    k2 = _noclick_drift(Γ .+ (dt / 2) .* k1, jumps)
    k3 = _noclick_drift(Γ .+ (dt / 2) .* k2, jumps)
    k4 = _noclick_drift(Γ .+ dt .* k3, jumps)
    G = Γ .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    s.Gamma = (G .- transpose(G)) ./ 2
    s
end

"""
    step!(s::MajoranaState, O::AbstractMatrix, J::MajoranaJumps, dt; rng=Random.default_rng())
    step!(s::MajoranaState, H::BdGHamiltonian, J::MajoranaJumps, dt; rng=Random.default_rng())

One MCWF step for monitored BdG dynamics: apply the unitary (covariance propagator
`O`, or `propagator(H, dt)`); then for the linear jump channels draw at most one
click over `dt` (rate `Σ N_m dt`, conditional state on click; no-click drift
otherwise) and finally apply any projective occupation monitors. The trajectory
ensemble reproduces the matching `MajoranaLindblad`.
"""
function step!(s::MajoranaState, O::AbstractMatrix, J::MajoranaJumps, dt::Real;
               rng::AbstractRNG=Random.default_rng())
    size(s.Gamma, 1) == J.twoL ||
        throw(ArgumentError("MajoranaState size $(size(s.Gamma, 1)) does not match jump set size $(J.twoL)"))
    evolve!(s, O)
    if !isempty(J.jumps)
        Γ = s.Gamma
        rates = Float64[first(_jump_rate_and_increment(Γ, ℓ)) for ℓ in J.jumps]
        total = sum(rates)
        if rand(rng) < total * dt                       # a click occurs
            r = rand(rng) * total
            idx = 1; acc = rates[1]
            while acc < r && idx < length(rates)
                idx += 1; acc += rates[idx]
            end
            _apply_majorana_jump!(s, J.jumps[idx])
        else                                             # no-click back-action
            _noclick_step!(s, J.jumps, dt)
        end
    end
    for (i, γ) in J.monitors
        rand(rng) < γ * dt && measure!(s, i; rng)
    end
    s
end

step!(s::MajoranaState, H::BdGHamiltonian, J::MajoranaJumps, dt::Real;
      rng::AbstractRNG=Random.default_rng()) =
    step!(s, propagator(H, dt), J, dt; rng)

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
# Apply the exact M = exp(α nᵢ) Gaussian filter to the covariance (renormalized state).
function _filter_occupation!(s::MajoranaState, i::Integer, α::Real)
    twoL = size(s.Gamma, 1)
    p = i; q = i + nmodes(s)
    Γ = s.Gamma
    G = Matrix{ComplexF64}(I, twoL, twoL) .- 1im .* Γ        # ⟨ωωᵀ⟩, Hermitian
    w = Γ[p, q]
    rp = G[p, :]; rq = G[q, :]
    X = rp * transpose(rq) .- rq * transpose(rp)
    G1 = (1im / 2) .* (X .+ conj.(X))                         # dG/dα|₀ ; X̄ = c_p c_qᵀ - c_q c_pᵀ
    T1 = 2w .* G .- 2 .* G1
    e = ones(twoL); e[p] = -1.0; e[q] = -1.0
    T2 = (e * transpose(e)) .* G                              # E G E = ε_a ε_b G_ab
    c = cosh(α / 2); sn = sinh(α / 2)
    Gp = (c^2 .* G .- c * sn .* T1 .+ sn^2 .* T2) ./ (cosh(α) - w * sinh(α))
    Gout = real.(1im .* (Gp .- Matrix{ComplexF64}(I, twoL, twoL)))
    s.Gamma = (Gout .- transpose(Gout)) ./ 2
    s
end

"""
    step_diffusive!(s::MajoranaState, monitors, dt; rng=Random.default_rng()) -> s

One quantum-state-diffusion step of continuous occupation monitoring. `monitors` is
an iterable of `(site, rate)` pairs; each monitored site applies the weak Gaussian
filter `exp(αᵢ nᵢ)` with `αᵢ = δWᵢ + (2⟨nᵢ⟩ − 1) γᵢ dt`, `δWᵢ ~ 𝒩(0, γᵢ dt)`
(the convention shared with the `SlaterState` [`step_diffusive!`](@ref)). Averaging
over the Wiener noise reproduces the dephasing Lindblad `D[√γ nᵢ]`, so the ensemble
matches `MajoranaLindblad(H; dephasing_ops=[(eᵢ, γ)])`.
"""
function step_diffusive!(s::MajoranaState, monitors, dt::Real; rng::AbstractRNG=Random.default_rng())
    L = nmodes(s)
    for (i, γ) in monitors
        1 ≤ Int(i) ≤ L || throw(ArgumentError("monitor site $i out of range 1:$L"))
        γf = Float64(γ)
        α = randn(rng) * sqrt(γf * dt) + (2 * density(s, Int(i)) - 1) * γf * dt
        _filter_occupation!(s, Int(i), α)
    end
    s
end
