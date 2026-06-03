#---------------------------------------------------------------------------------------------------
# Majorana / BdG covariance states
#---------------------------------------------------------------------------------------------------
export MajoranaState, covariance_matrix
export fermion_correlations, normal_correlation, anomalous_correlation

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

mutable struct MajoranaState{T<:Real} <: AbstractGaussianState{T}
    Gamma::Matrix{T}

    function MajoranaState{T}(Gamma::Matrix{T}; check::Bool=true, atol::Real=1e-10) where {T<:Real}
        _validate_majorana_covariance(Gamma; check, atol)
        new{T}((Gamma - transpose(Gamma)) / 2)
    end
end

MajoranaState(Gamma::AbstractMatrix; check::Bool=true, atol::Real=1e-10) =
    MajoranaState{Float64}(Matrix{Float64}(Gamma); check, atol)

function MajoranaState(occ::AbstractVector{<:Integer})
    all(x -> x == 0 || x == 1, occ) ||
        throw(ArgumentError("occupation vector entries must be 0 or 1"))
    L = length(occ)
    D = Diagonal(1 .- 2 .* Float64.(occ))
    Z = zeros(Float64, L, L)
    MajoranaState([Z D; -D Z]; check=false)
end

MajoranaState(s::SlaterState) = MajoranaState(CorrelationState(s))

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

covariance_matrix(s::MajoranaState) = copy(s.Gamma)

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

normal_correlation(s::MajoranaState) = first(fermion_correlations(s))
anomalous_correlation(s::MajoranaState) = last(fermion_correlations(s))
