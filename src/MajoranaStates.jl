#---------------------------------------------------------------------------------------------------
# Majorana / BdG covariance states
#---------------------------------------------------------------------------------------------------
export MajoranaState, covariance_matrix
export fermion_correlations, normal_correlation, anomalous_correlation
# `ispure`, `nmodes` are exported from States.jl; methods for MajoranaState live here.

function _validate_majorana_covariance(Gamma::AbstractMatrix; check::Bool=true, atol::Real=1e-10)
    size(Gamma, 1) == size(Gamma, 2) ||
        throw(ArgumentError("Majorana covariance matrix must be square, got $(size(Gamma))"))
    iseven(size(Gamma, 1)) ||
        throw(ArgumentError("Majorana covariance matrix size must be even, got $(size(Gamma, 1))"))

    G = Matrix{Float64}(Gamma)
    all(isfinite, G) ||
        throw(ArgumentError("Majorana covariance matrix entries must be finite"))

    if check
        isapprox(G, -transpose(G); atol) ||
            throw(ArgumentError("Majorana covariance matrix must be antisymmetric"))
        vals = real.(eigvals(Hermitian(1im * G)))
        maximum(abs.(vals); init=0.0) ≤ 1 + sqrt(atol) ||
            throw(ArgumentError("Majorana covariance matrix has unphysical spectrum outside [-1, 1]"))
    end
end

"""
    MajoranaState{T} <: AbstractGaussianState{T}

General (possibly mixed, possibly paired) fermionic Gaussian state on `L` modes,
stored by its real antisymmetric `2L×2L` Majorana covariance matrix

    Γ_ab = (i/2)⟨[ωₐ, ω_b]⟩,   ω = [x₁…x_L, p₁…p_L],  xⱼ = cⱼ+c⁺ⱼ,  pⱼ = i(cⱼ-c⁺ⱼ).

`T<:AbstractFloat` is the covariance element type (`Float64` by default).
"""
mutable struct MajoranaState{T<:AbstractFloat} <: AbstractGaussianState{T}
    Gamma::Matrix{T}

    function MajoranaState{T}(Gamma::Matrix{T}; check::Bool=true, atol::Real=1e-10) where {T<:AbstractFloat}
        _validate_majorana_covariance(Gamma; check, atol)
        new{T}((Gamma - transpose(Gamma)) / 2)
    end
end

"""
    MajoranaState(Γ::AbstractMatrix; check=true, atol=1e-10)

Wrap a Majorana covariance matrix `Γ` (real, antisymmetric, spectrum of `iΓ` in
`[-1,1]`). A floating-point element type (`Float32`/`Float64`/`BigFloat`) is
preserved; integer/other inputs are promoted to `Float64`. With `check=false` the
physicality validation is skipped (use only for covariances you trust).
"""
MajoranaState(Gamma::AbstractMatrix{T}; check::Bool=true, atol::Real=1e-10) where {T<:AbstractFloat} =
    MajoranaState{T}(Matrix{T}(Gamma); check, atol)
MajoranaState(Gamma::AbstractMatrix; check::Bool=true, atol::Real=1e-10) =
    MajoranaState{Float64}(Matrix{Float64}(Gamma); check, atol)

"""
    MajoranaState(occ::AbstractVector{<:Integer})

Product (Fock) state from an occupation vector `occ` of 0/1 entries:
`Γ[i, i+L] = 1 - 2 occᵢ`.
"""
function MajoranaState(occ::AbstractVector{<:Integer})
    all(x -> x == 0 || x == 1, occ) ||
        throw(ArgumentError("occupation vector entries must be 0 or 1"))
    L = length(occ)
    D = Diagonal(1 .- 2 .* Float64.(occ))
    Z = zeros(Float64, L, L)
    MajoranaState([Z D; -D Z]; check=false)
end

"""    MajoranaState(s::SlaterState) — covariance form of a pure number-conserving state."""
MajoranaState(s::SlaterState) = MajoranaState(CorrelationState(s))

"""
    MajoranaState(s::CorrelationState; check=true, atol=1e-10)

Covariance form of a (possibly mixed) number-conserving state: builds Γ from the
correlation matrix `Cᵢⱼ = ⟨c⁺ᵢcⱼ⟩` with zero anomalous part `Fᵢⱼ = ⟨cᵢcⱼ⟩ = 0`.
"""
MajoranaState(s::CorrelationState; check::Bool=true, atol::Real=1e-10) =
    MajoranaState(_majorana_covariance(correlation_matrix(s), zeros(ComplexF64, nmodes(s), nmodes(s)));
                  check, atol)

# Standard Dirac-correlation → Majorana covariance map (inverse of `fermion_correlations`).
# Convention: C = ⟨c⁺ᵢcⱼ⟩ (Hermitian), F = ⟨cᵢcⱼ⟩ (antisymmetric); covariance
# Γ_ab = (i/2)⟨[ωₐ,ω_b]⟩ in the basis ω = [x₁…x_L, p₁…p_L], xⱼ = cⱼ+c⁺ⱼ, pⱼ = i(cⱼ-c⁺ⱼ).
# Verified against exact diagonalization (Γ from explicit Majorana operators).
function _majorana_covariance(C::AbstractMatrix, F::AbstractMatrix)
    L = size(C, 1)
    Cr, Ci = real.(C), imag.(C)
    Fr, Fi = real.(F), imag.(F)
    Eye = Matrix{Float64}(I, L, L)
    G11 = -2 .* Ci .- 2 .* Fi
    G12 = Eye .- 2 .* Cr .- 2 .* Fr
    G21 = 2 .* Cr .- Eye .- 2 .* Fr
    G22 = -2 .* Ci .+ 2 .* Fi
    Matrix{Float64}([G11 G12; G21 G22])
end

nmodes(s::MajoranaState) = size(s.Gamma, 1) ÷ 2
Base.eltype(::MajoranaState{T}) where {T} = T
Base.copy(s::MajoranaState) = MajoranaState(copy(s.Gamma); check=false)

"""
    ispure(s::MajoranaState; tol=1e-8) -> Bool

Whether the covariance state is pure: all eigenvalues of `iΓ` are `≈ ±1`
(equivalently `Γ² ≈ -I`).
"""
function ispure(s::MajoranaState; tol::Real=1e-8)
    ν = real.(eigvals(Hermitian(1im .* s.Gamma)))
    all(x -> abs(abs(x) - 1) < tol, ν)
end

"""
    covariance_matrix(s::MajoranaState) -> Matrix

Return a copy of the Majorana covariance matrix
`Γ_ab = (i/2)⟨[ω_a,ω_b]⟩` in the basis
`ω = [x_1,...,x_L,p_1,...,p_L]`.
"""
covariance_matrix(s::MajoranaState) = copy(s.Gamma)

"""
    fermion_correlations(s::MajoranaState) -> (C, F)

Convert the Majorana covariance to Dirac correlations using the package
conventions `C[i,j] = ⟨c_i† c_j⟩` and `F[i,j] = ⟨c_i c_j⟩`.
"""
function fermion_correlations(s::MajoranaState)
    G = s.Gamma
    L = nmodes(s)
    G11 = view(G, 1:L, 1:L)
    G12 = view(G, 1:L, L+1:2L)
    G21 = view(G, L+1:2L, 1:L)
    G22 = view(G, L+1:2L, L+1:2L)
    C = (G21 .- G12 .- 1im .* G11 .- 1im .* G22) ./ 4 .+ Matrix{ComplexF64}(I, L, L) ./ 2
    F = (.-G21 .- G12 .- 1im .* G11 .+ 1im .* G22) ./ 4
    Matrix{ComplexF64}(C), Matrix{ComplexF64}(F)
end

"""
    normal_correlation(s::MajoranaState) -> Matrix{ComplexF64}

Return `C[i,j] = ⟨c_i† c_j⟩` from a Majorana covariance state.
"""
normal_correlation(s::MajoranaState) = first(fermion_correlations(s))

"""
    anomalous_correlation(s::MajoranaState) -> Matrix{ComplexF64}

Return `F[i,j] = ⟨c_i c_j⟩` from a Majorana covariance state.
"""
anomalous_correlation(s::MajoranaState) = last(fermion_correlations(s))
