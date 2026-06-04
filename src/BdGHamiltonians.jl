#---------------------------------------------------------------------------------------------------
# BdG / Majorana quadratic Hamiltonians
#---------------------------------------------------------------------------------------------------
export BdGHamiltonian, groundstate, quasiparticle_energies

"""
    BdGHamiltonian(K::AbstractMatrix; check=true, atol=1e-10)
    BdGHamiltonian(A::AbstractMatrix, B::AbstractMatrix; check=true, atol=1e-10)
    BdGHamiltonian(H::QuadraticHamiltonian)

General quadratic (BdG/Majorana) Hamiltonian. The one-argument constructor accepts
the real antisymmetric Majorana covariance generator `K`, for which
`Γ(t) = exp(K*t) Γ(0) exp(K*t)'`. The two-block constructor accepts a Hermitian
hopping block `A` and antisymmetric pairing block `B` in
`Σ Aᵢⱼ cᵢ†cⱼ + 1/2 Σ(Bᵢⱼ cᵢ†cⱼ† + h.c.)`.
"""
mutable struct BdGHamiltonian{T<:Real} <: AbstractQuadraticHamiltonian
    K::Matrix{T}
    cache::Union{Nothing,Tuple{Float64,Matrix{T}}}
end

function _check_square_even_matrix(M::AbstractMatrix, label::String)
    size(M, 1) == size(M, 2) ||
        throw(ArgumentError("$label must be square, got $(size(M))"))
    iseven(size(M, 1)) ||
        throw(ArgumentError("$label size must be even, got $(size(M, 1))"))
end

function BdGHamiltonian(K::AbstractMatrix; check::Bool=true, atol::Real=1e-10)
    _check_square_even_matrix(K, "BdG Majorana generator")
    Kmat = Matrix{Float64}(K)
    all(isfinite, Kmat) ||
        throw(ArgumentError("BdG Majorana generator entries must be finite"))
    if check
        isapprox(Kmat, -transpose(Kmat); atol) ||
            throw(ArgumentError("BdG Majorana generator must be antisymmetric"))
    end
    BdGHamiltonian{Float64}((Kmat - transpose(Kmat)) / 2, nothing)
end

# Majorana basis ω = [x₁…x_L, p₁…p_L] with xⱼ = cⱼ+c⁺ⱼ, pⱼ = i(cⱼ-c⁺ⱼ). The map
# (c; c⁺) → ω is ω = Ω (c; c⁺) with Ω = [[I, I],[iI, -iI]].
_omega(L)     = [Matrix{ComplexF64}(I, L, L)  Matrix{ComplexF64}(I, L, L);
                 im .* Matrix{ComplexF64}(I, L, L)  (-im) .* Matrix{ComplexF64}(I, L, L)]
_omega_inv(L) = 0.5 .* [Matrix{ComplexF64}(I, L, L)  (-im) .* Matrix{ComplexF64}(I, L, L);
                        Matrix{ComplexF64}(I, L, L)  im .* Matrix{ComplexF64}(I, L, L)]

# Covariance evolution generator K from Dirac BdG blocks: Ĥ = Σ Aᵢⱼc⁺ᵢcⱼ +
# ½Σ(Bᵢⱼc⁺ᵢc⁺ⱼ + h.c.). Built from the Nambu Hamiltonian ℋ = [A B; -B̄ -Aᵀ] via
# the Heisenberg generator K = Re[Ω(-iℋ)Ω⁻¹], so Γ(dt) = exp(K dt) Γ exp(K dt)ᵀ.
# Verified against CorrelationState evolution (real and complex A) and exact
# diagonalization (standard Majorana convention).
function _bdg_blocks_to_K(A::AbstractMatrix, B::AbstractMatrix)
    size(A) == size(B) ||
        throw(ArgumentError("BdG blocks A and B must have the same size, got $(size(A)) and $(size(B))"))
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("BdG block A must be square, got $(size(A))"))
    isapprox(A, A'; atol=1e-10) ||
        throw(ArgumentError("BdG block A must be Hermitian"))
    isapprox(B, -transpose(B); atol=1e-10) ||
        throw(ArgumentError("BdG pairing block B must be antisymmetric"))
    L = size(A, 1)
    ℋ = [Matrix{ComplexF64}(A)  Matrix{ComplexF64}(B); -conj(B)  -transpose(A)]
    K = real.(_omega(L) * (-im .* ℋ) * _omega_inv(L))
    Matrix{Float64}((K .- transpose(K)) ./ 2)
end

BdGHamiltonian(A::AbstractMatrix, B::AbstractMatrix; check::Bool=true, atol::Real=1e-10) =
    BdGHamiltonian(_bdg_blocks_to_K(A, B); check, atol)

BdGHamiltonian(H::QuadraticHamiltonian) =
    BdGHamiltonian(_bdg_blocks_to_K(Matrix(H.h), zeros(ComplexF64, nmodes(H), nmodes(H))); check=false)

nmodes(H::BdGHamiltonian) = size(H.K, 1) ÷ 2
Base.Matrix(H::BdGHamiltonian) = copy(H.K)

function propagator(H::BdGHamiltonian, dt::Real)
    if H.cache !== nothing && H.cache[1] == Float64(dt)
        return H.cache[2]
    end
    O = exp(Float64(dt) .* H.K)
    H.cache = (Float64(dt), O)
    O
end

function evolve!(s::MajoranaState, O::AbstractMatrix)
    size(O) == size(s.Gamma) ||
        throw(ArgumentError("Majorana propagator size $(size(O)) does not match covariance size $(size(s.Gamma))"))
    Omat = Matrix{Float64}(O)
    G = Omat * s.Gamma * transpose(Omat)
    s.Gamma = (G - transpose(G)) / 2
    s
end

evolve!(s::MajoranaState, H::BdGHamiltonian, dt::Real) = evolve!(s, propagator(H, dt))
evolve(s::MajoranaState, args...) = evolve!(copy(s), args...)

#---------------------------------------------------------------------------------------------------
# Quasiparticle diagonalization: ground / thermal states
#
# Recover the Dirac BdG blocks (A, B) from the covariance generator K = majoranaform(A, B),
# build the Nambu Hamiltonian ℋ = [A B; -B̄ -Aᵀ], and read off the ground/thermal
# correlations (C, F) in the package convention. Validated against exact diagonalization.
#---------------------------------------------------------------------------------------------------
# Recover the Nambu Hamiltonian ℋ = [A B; -B̄ -Aᵀ] from the covariance generator K
# (inverse of `_bdg_blocks_to_K`): ℋ = i Ω⁻¹ K Ω.
function _nambu_hamiltonian(H::BdGHamiltonian)
    L = nmodes(H)
    M = im .* (_omega_inv(L) * H.K * _omega(L))
    Hermitian((M .+ M') ./ 2)
end

# Nambu correlation ⟨ΨΨ†⟩ with mode weights `weight(ε)`; extract (C, F) → covariance.
function _state_from_nambu(H::BdGHamiltonian, weight)
    L = nmodes(H)
    E, W = eigen(_nambu_hamiltonian(H))
    Gn = W * Diagonal(weight.(E)) * W'
    C = (Gn[L+1:2L, L+1:2L] .+ Gn[L+1:2L, L+1:2L]') ./ 2          # C = ⟨c⁺ᵢcⱼ⟩, Hermitian
    Fraw = Gn[1:L, L+1:2L]
    F = (Fraw .- transpose(Fraw)) ./ 2                            # F = ⟨cᵢcⱼ⟩, antisymmetric
    MajoranaState(_majorana_covariance(C, F))
end

"""
    groundstate(H::BdGHamiltonian) -> MajoranaState
    groundstate(H::QuadraticHamiltonian) -> MajoranaState

Ground state of the quadratic Hamiltonian as a Majorana covariance state, via
Bogoliubov/Nambu diagonalization (occupy the positive-energy Nambu modes). For a
number-conserving `H` this is the Fermi sea (all negative single-particle orbitals
filled). Exact zero modes are degenerate; this routine applies the package's
`ε > 0` occupation convention and may fail validation for some degenerate BdG
systems. Detune or split zero modes when a definite parity sector is required.
"""
groundstate(H::BdGHamiltonian) = _state_from_nambu(H, ε -> ε > 0 ? 1.0 : 0.0)
groundstate(H::QuadraticHamiltonian) = groundstate(BdGHamiltonian(H))

"""
    thermalstate(H::BdGHamiltonian; β) -> MajoranaState

Gibbs state `ρ ∝ exp(-β Ĥ)` of a BdG Hamiltonian as a Majorana covariance state.
When the BdG spectrum has no exact zero modes, `β → ∞` recovers
[`groundstate`](@ref). Exact zero modes remain half occupied in the thermal
limit, so this need not match a parity-selected ground state.
"""
thermalstate(H::BdGHamiltonian; β::Real) = _state_from_nambu(H, ε -> 1 / (1 + exp(-β * ε)))

"""
    quasiparticle_energies(H::BdGHamiltonian) -> Vector{Float64}

The non-negative Bogoliubov quasiparticle energies (the upper half of the
particle-hole-symmetric Nambu spectrum), sorted ascending.
"""
function quasiparticle_energies(H::BdGHamiltonian)
    L = nmodes(H)
    L == 0 && return Float64[]

    E = sort(real.(eigvals(_nambu_hamiltonian(H))))
    E[end-L+1:end]
end
