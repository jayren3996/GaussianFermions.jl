#---------------------------------------------------------------------------------------------------
# Quantum trajectories for Majorana/BdG covariance states
#
# Building blocks for monitored BdG dynamics. Projective occupation measurement is the
# fundamental conditional update: measuring mode i selects the 2×2 Majorana block
# (xᵢ, pᵢ) to the definite-occupation covariance σ and conditions the rest by the
# fermionic-Gaussian Schur complement. Verified against the number-conserving
# `SlaterState` measurement.
#---------------------------------------------------------------------------------------------------
export measure!, step!

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
