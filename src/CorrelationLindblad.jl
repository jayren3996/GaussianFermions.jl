#---------------------------------------------------------------------------------------------------
# Deterministic correlation-matrix Lindblad dynamics
#---------------------------------------------------------------------------------------------------
export CorrelationLindblad, lindblad_rhs, steadystate

function steadystate end

mutable struct CorrelationLindblad
    h::Hermitian{ComplexF64,Matrix{ComplexF64}}
    damping::Matrix{ComplexF64}
    source::Matrix{ComplexF64}
    dephasing::Vector{Tuple{Float64,Matrix{ComplexF64}}}
    cache::Union{Nothing,Tuple{UInt,Matrix{ComplexF64},Vector{ComplexF64}}}
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

_dephasing_entry(op, L::Integer) =
    throw(ArgumentError("dephasing entries must be (mode_or_projector, gamma)"))

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
    size(Cmat) == size(Liouv.h) ||
        throw(ArgumentError("correlation matrix size $(size(Cmat)) does not match system size $(size(Liouv.h))"))
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

function _hash_matrix(A::AbstractMatrix, seed::UInt)
    h = hash(size(A), seed)
    for value in A
        h = hash(value, h)
    end
    h
end

function _generator_fingerprint(Liouv::CorrelationLindblad)
    h = hash(:CorrelationLindbladGenerator, zero(UInt))
    h = _hash_matrix(Matrix(Liouv.h), h)
    h = _hash_matrix(Liouv.damping, h)
    h = _hash_matrix(Liouv.source, h)
    h = hash(length(Liouv.dephasing), h)
    for (γ, Q) in Liouv.dephasing
        h = hash(γ, h)
        h = _hash_matrix(Q, h)
    end
    h
end

function _affine_generator(Liouv::CorrelationLindblad)
    fingerprint = _generator_fingerprint(Liouv)
    if Liouv.cache !== nothing
        cached_fingerprint, M, b = Liouv.cache
        cached_fingerprint == fingerprint && return M, b
    end

    L = size(Liouv.h, 1)
    n = L * L
    Czero = zeros(ComplexF64, L, L)
    b = vec(lindblad_rhs(Liouv, Czero))
    M = zeros(ComplexF64, n, n)
    for k in 1:n
        E = zeros(ComplexF64, L, L)
        E[k] = 1
        M[:, k] .= vec(lindblad_rhs(Liouv, E)) .- b
    end
    Liouv.cache = (fingerprint, M, b)
    M, b
end

function _hermitian_correlation(C::AbstractMatrix)
    Cs = (Matrix{ComplexF64}(C) + Matrix{ComplexF64}(C)') / 2
    Hermitian(Cs)
end

function evolve!(s::CorrelationState, Liouv::CorrelationLindblad, dt::Real; method::Symbol=:expm)
    method == :expm || throw(ArgumentError("unsupported CorrelationLindblad evolution method: $method"))
    M, b = _affine_generator(Liouv)
    n = length(b)
    A = zeros(ComplexF64, n + 1, n + 1)
    A[1:n, 1:n] .= M
    A[1:n, n + 1] .= b
    y = exp(dt .* A) * vcat(vec(correlation_matrix(s)), one(ComplexF64))
    L = nmodes(s)
    s.C = _hermitian_correlation(reshape(y[1:n], L, L))
    s
end

evolve(s::CorrelationState, Liouv::CorrelationLindblad, dt::Real; kwargs...) =
    evolve!(copy(s), Liouv, dt; kwargs...)

function steadystate(Liouv::CorrelationLindblad; atol::Real=1e-10, rtol::Real=1e-8, check_unique::Bool=true)
    M, b = _affine_generator(Liouv)
    if check_unique
        svals = svdvals(M)
        singular_threshold = rtol * maximum(svals)
        if minimum(svals) ≤ singular_threshold
            throw(ArgumentError("CorrelationLindblad steady state is not unique; the dense generator is singular"))
        end
    end

    x = try
        -(M \ b)
    catch err
        throw(ArgumentError("CorrelationLindblad steady-state solve failed: $(err)"))
    end

    L = size(Liouv.h, 1)
    C = Matrix(_hermitian_correlation(reshape(x, L, L)))
    residual = norm(lindblad_rhs(Liouv, C))
    residual ≤ atol + rtol * (norm(M) * norm(x) + norm(b)) ||
        throw(ArgumentError("CorrelationLindblad steady-state residual $residual exceeds tolerance"))

    vals = real.(eigvals(Hermitian(C)))
    all(vals .≥ -sqrt(atol)) && all(vals .≤ 1 + sqrt(atol)) ||
        throw(ArgumentError("CorrelationLindblad steady state is outside the physical occupation range"))

    CorrelationState(Hermitian(C))
end
