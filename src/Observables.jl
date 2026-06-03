#---------------------------------------------------------------------------------------------------
# Observables and entanglement diagnostics
#
# Everything is computed from the single-particle correlation matrix C (or its
# restricted block C_A), so the same code works for pure (`SlaterState`) and mixed
# (`CorrelationState`) states. `SlaterState` adds a faster SVD-of-B path for the
# von Neumann entropy.
#---------------------------------------------------------------------------------------------------
export density, density_profile, particle_number, number_variance, purity, parity
export correlation_profile
export entanglement_entropy, renyi_entropy, mutual_information, tripartite_information
export entanglement_spectrum, entanglement_hamiltonian_spectrum

# robust 0·log0 = 0
_xlogx(x::Real) = x ≤ 0 ? zero(x) : x * log(x)
# binary Shannon entropy of an occupation ν ∈ [0,1]
_binary_shannon(ν::Real) = -_xlogx(ν) - _xlogx(1 - ν)

"""
    _occupation_spectrum(s, A) -> Vector{Float64}

Single-particle entanglement spectrum of region `A`: eigenvalues of the restricted
correlation block `C_A`, clamped to `[0,1]` (kills `-1e-16` / `1+1e-16`).
"""
function _occupation_spectrum(s::NumberConservingGaussianState, A)
    ν = eigvals(Hermitian(correlation_matrix(s, collect(A))))
    clamp.(real.(ν), 0.0, 1.0)
end

#---------------------------------------------------------------------------------------------------
# Local / global one-body observables
#---------------------------------------------------------------------------------------------------
"""
    density(s)      -> Vector{Float64}
    density(s, i)   -> Float64

Site occupations `⟨nᵢ⟩ = ⟨c⁺ᵢ cᵢ⟩`.
"""
density(s::NumberConservingGaussianState) = [real(correlation(s, i, i)) for i in 1:nmodes(s)]
density(s::NumberConservingGaussianState, i::Integer) = real(correlation(s, i, i))
# occupation ⟨nₘ⟩ = ⟨d†d⟩ of a quasi-mode d = Σₐ Vₐ c_{Iₐ} (pure SlaterState)
density(s::SlaterState, qm::QuasiMode) = (p = inner(qm, s.B); real(dot(p, p)))

"""
    density_profile(s) -> Vector{Float64}

Alias for [`density`](@ref) (occupation as a function of site).
"""
density_profile(s::NumberConservingGaussianState) = density(s)

"""
    particle_number(s)      -> total ⟨N⟩
    particle_number(s, A)   -> ⟨N_A⟩

Mean particle number. For a `SlaterState` the total is the exact integer `N`.
"""
particle_number(s::SlaterState) = size(s.B, 2)
particle_number(s::CorrelationState) = real(tr(s.C))
particle_number(s::NumberConservingGaussianState, A) = sum(real(correlation(s, i, i)) for i in A)

"""
    correlation_profile(s, i) -> Vector

The correlations `⟨c⁺ᵢ cⱼ⟩` from a reference site `i` to every site `j`.
"""
correlation_profile(s::NumberConservingGaussianState, i::Integer) =
    [correlation(s, i, j) for j in 1:nmodes(s)]

#---------------------------------------------------------------------------------------------------
# Entanglement entropy and Rényi entropies
#---------------------------------------------------------------------------------------------------
"""
    entanglement_entropy(s, A) -> Float64

Von Neumann entanglement entropy of region `A`,
`S(A) = -∑ₖ [νₖ log νₖ + (1-νₖ) log(1-νₖ)]`, where `{νₖ}` are the eigenvalues of
`C_A`.
"""
function entanglement_entropy(s::NumberConservingGaussianState, A::AbstractVector{<:Integer})
    sum(_binary_shannon, _occupation_spectrum(s, A))
end
# fast pure-state path: νₖ = σₖ² with σ = svdvals(B[A,:]); missing eigenvalues are 0/1
# (zero contribution), so the sum is complete. Valid only for a pure global state.
function entanglement_entropy(s::SlaterState, A::AbstractVector{<:Integer})
    σ = svdvals(s.B[A, :])
    acc = 0.0
    for σk in σ
        acc += _binary_shannon(clamp(σk^2, 0.0, 1.0))
    end
    acc
end

"""
    renyi_entropy(s, A; α=2) -> Float64

Rényi-α entanglement entropy `Sα(A) = (1/(1-α)) ∑ₖ log(νₖ^α + (1-νₖ)^α)`.
`α == 1` returns the von Neumann entropy; `α == Inf` the min-entropy
`-∑ₖ log max(νₖ, 1-νₖ)`.
"""
function renyi_entropy(s::NumberConservingGaussianState, A::AbstractVector{<:Integer}; α::Real=2)
    α == 1 && return entanglement_entropy(s, A)
    ν = _occupation_spectrum(s, A)
    isinf(α) && return -sum(log(max(νk, 1 - νk)) for νk in ν)
    (1 / (1 - α)) * sum(log(νk^α + (1 - νk)^α) for νk in ν)
end

"""
    mutual_information(s, A, B; α=1) -> Float64

`I(A:B) = S(A) + S(B) - S(A∪B)` (Rényi version uses `Sα`).
"""
function mutual_information(s::NumberConservingGaussianState,
                           A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer}; α::Real=1)
    f(R) = renyi_entropy(s, R; α)
    f(A) + f(B) - f(vcat(A, B))
end

"""
    tripartite_information(s, A, B, C; α=1) -> Float64

`I₃ = S(A)+S(B)+S(C) - S(AB) - S(AC) - S(BC) + S(ABC)` — the order parameter for
measurement-induced transitions. `A`, `B`, `C` must be disjoint.
"""
function tripartite_information(s::NumberConservingGaussianState,
                               A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer},
                               C::AbstractVector{<:Integer}; α::Real=1)
    S(R) = renyi_entropy(s, R; α)
    S(A) + S(B) + S(C) - S(vcat(A, B)) - S(vcat(A, C)) - S(vcat(B, C)) + S(vcat(A, B, C))
end

#---------------------------------------------------------------------------------------------------
# Charge fluctuations, purity, parity, spectra
#---------------------------------------------------------------------------------------------------
"""
    number_variance(s, A) -> Float64

Charge fluctuation `Var(N_A) = ∑ₖ νₖ(1-νₖ)`.
"""
number_variance(s::NumberConservingGaussianState, A::AbstractVector{<:Integer}) =
    sum(νk * (1 - νk) for νk in _occupation_spectrum(s, A))

"""
    purity(s, A=1:nmodes(s)) -> Float64

`Tr ρ_A² = ∏ₖ (νₖ² + (1-νₖ)²)`. Equals 1 for a pure global state.
"""
function purity(s::NumberConservingGaussianState, A=1:nmodes(s))
    prod(νk^2 + (1 - νk)^2 for νk in _occupation_spectrum(s, A))
end

"""
    parity(s, A=1:nmodes(s)) -> Float64

Fermion parity `⟨(-1)^{N_A}⟩ = det(I - 2 C_A)`.
"""
function parity(s::NumberConservingGaussianState, A=1:nmodes(s))
    real(det(I - 2 * correlation_matrix(s, collect(A))))
end

"""
    entanglement_spectrum(s, A) -> Vector{Float64}

Single-particle entanglement spectrum: the occupations `νₖ ∈ [0,1]` of `C_A`
(sorted ascending).
"""
entanglement_spectrum(s::NumberConservingGaussianState, A::AbstractVector{<:Integer}) =
    _occupation_spectrum(s, A)

"""
    entanglement_hamiltonian_spectrum(s, A) -> Vector{Float64}

Entanglement-Hamiltonian single-particle levels `εₖ = log((1-νₖ)/νₖ)` (±Inf for
`νₖ = 0` or `1`).
"""
entanglement_hamiltonian_spectrum(s::NumberConservingGaussianState, A::AbstractVector{<:Integer}) =
    [log((1 - νk) / νk) for νk in _occupation_spectrum(s, A)]
