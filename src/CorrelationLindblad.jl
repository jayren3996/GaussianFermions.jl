#---------------------------------------------------------------------------------------------------
# Deterministic correlation-matrix Lindblad dynamics
#---------------------------------------------------------------------------------------------------
export CorrelationLindblad, lindblad_rhs, steadystate

mutable struct CorrelationLindblad
    h::Hermitian{ComplexF64,Matrix{ComplexF64}}
    damping::Matrix{ComplexF64}
    source::Matrix{ComplexF64}
    dephasing::Vector{Tuple{Float64,Matrix{ComplexF64}}}
    cache::Union{Nothing,Tuple{Matrix{ComplexF64},Vector{ComplexF64}}}
end

function CorrelationLindblad(h::AbstractMatrix; loss_ops=[], gain_ops=[], dephasing_ops=[])
    H = Hermitian(Matrix{ComplexF64}(h))
    L = size(H, 1)
    damping = zeros(ComplexF64, L, L)
    source = zeros(ComplexF64, L, L)

    for op in loss_ops
        Γ = _linear_rate_matrix(op, L)
        damping .+= Γ
    end
    for op in gain_ops
        Γ = _linear_rate_matrix(op, L)
        damping .+= Γ
        source .+= conj(Γ)
    end

    dephasing = [_dephasing_entry(op, L) for op in dephasing_ops]
    CorrelationLindblad(H, damping, source, dephasing, nothing)
end

CorrelationLindblad(H::QuadraticHamiltonian; kwargs...) = CorrelationLindblad(Matrix(H.h); kwargs...)

function _dense_mode_vector(v::AbstractVector, L::Integer)
    length(v) == L || throw(ArgumentError("mode vector length $(length(v)) does not match system size $L"))
    Vector{ComplexF64}(v)
end

function _linear_rate_matrix(op::AbstractVector, L::Integer)
    w = _dense_mode_vector(op, L)
    w * w'
end

function _linear_rate_matrix(qm::QuasiMode, L::Integer)
    qm.L == L || throw(ArgumentError("QuasiMode length $(qm.L) does not match system size $L"))
    v = Vector{ComplexF64}(vector(qm))
    v * v'
end

function _dephasing_entry(op::Tuple, L::Integer)
    length(op) == 2 || throw(ArgumentError("dephasing entries must be (mode_or_projector, gamma)"))
    raw, γ = op
    γf = Float64(γ)
    γf ≥ 0 || throw(ArgumentError("dephasing rate must be non-negative"))
    Q = _dephasing_projector(raw, L)
    (γf, Q)
end

function _dephasing_projector(v::AbstractVector, L::Integer)
    mode = _dense_mode_vector(v, L)
    conj(mode * mode')
end

function _dephasing_projector(P::AbstractMatrix, L::Integer)
    size(P) == (L, L) || throw(ArgumentError("dephasing projector size $(size(P)) does not match ($L, $L)"))
    Matrix{ComplexF64}(P)
end

function _dephasing_projector(qm::QuasiMode, L::Integer)
    qm.L == L || throw(ArgumentError("QuasiMode length $(qm.L) does not match system size $L"))
    v = Vector{ComplexF64}(vector(qm))
    conj(v * v')
end

function lindblad_rhs(Liouv::CorrelationLindblad, C::AbstractMatrix)
    Cmat = Matrix{ComplexF64}(C)
    h = Matrix(Liouv.h)
    dC = im .* (conj(h) * Cmat) .- im .* (Cmat * transpose(h))
    dC .+= Liouv.source
    dC .-= 0.5 .* (conj(Liouv.damping) * Cmat .+ Cmat * transpose(Liouv.damping))
    for (γ, Q) in Liouv.dephasing
        dC .-= (0.5 * γ) .* (Q * Cmat .+ Cmat * Q .- 2 .* (Q * Cmat * Q))
    end
    dC
end

lindblad_rhs(Liouv::CorrelationLindblad, s::CorrelationState) =
    lindblad_rhs(Liouv, correlation_matrix(s))
