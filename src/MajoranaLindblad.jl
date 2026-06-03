#---------------------------------------------------------------------------------------------------
# Deterministic Majorana/BdG covariance-matrix Lindblad dynamics
#
# The covariance Γ (real antisymmetric, Γ_ab = (i/2)⟨[ω_a, ω_b]⟩) of a general
# fermionic Gaussian state evolves under a quasi-free Lindbladian as
#
#   ∂ₜΓ = K Γ − Γ K                              (Hamiltonian, K = BdG generator)
#       − 2(Bᵣ Γ + Γ Bᵣ) + 4 Bᵢ                  (linear jumps,  B = Σ lₘ lₘ†)
#       + Σₖ[ ½(Mₖ² Γ + Γ Mₖ²) − Mₖ Γ Mₖ ]       (quadratic/dephasing jumps)
#
# Bᵣ/Bᵢ are the real/imaginary parts of the accumulated Hermitian bath matrix
# B = Σ lₘ lₘ† built from the 2L-component Majorana jump vectors lₘ; Mₖ are real
# antisymmetric Majorana quadratic forms. Number-conserving channels reproduce
# `CorrelationLindblad` exactly; linear jumps with a creation component generate
# anomalous (pairing) correlations. Unlike `CorrelationLindblad` this is the full
# BdG (number-non-conserving) generator.
#---------------------------------------------------------------------------------------------------
export MajoranaLindblad, majorana_lindblad_rhs

mutable struct MajoranaLindblad
    K::Matrix{Float64}              # 2L×2L antisymmetric Hamiltonian covariance generator
    Br::Matrix{Float64}            # Re(Σ lₘ lₘ†)  (symmetric)
    Bi::Matrix{Float64}            # Im(Σ lₘ lₘ†)  (antisymmetric)
    quad::Vector{Matrix{Float64}}  # quadratic jump matrices Mₖ (real antisymmetric)
    cache::Union{Nothing,Tuple{UInt,Matrix{Float64},Vector{Float64}}}
end

#---------------------------------------------------------------------------------------------------
# Majorana-form helpers (shared convention with BdGHamiltonians.jl)
#---------------------------------------------------------------------------------------------------
# Majorana quadratic form of a Dirac single-particle matrix A (Hermitian or
# antisymmetric); equals the legacy `majoranaform(A)` algebra.
_majoranaform_block(A::AbstractMatrix) =
    (AR = real.(A); AI = imag.(A); Matrix{Float64}([-AI AR; -AR -AI]))

# Majorana jump vector of a general linear Dirac jump L = Σᵢ aᵢ cᵢ + bᵢ cᵢ⁺,
# in the basis cᵢ = (xᵢ − i pᵢ)/2, ω = [x₁…x_L, p₁…p_L].
function _majorana_jump_vector(annih::AbstractVector, creat::AbstractVector, L::Integer)
    l = zeros(ComplexF64, 2L)
    @views l[1:L]      .= (annih .+ creat) ./ 2
    @views l[L+1:2L]   .= (im .* (creat .- annih)) ./ 2
    l
end

#---------------------------------------------------------------------------------------------------
# Constructors
#---------------------------------------------------------------------------------------------------
"""
    MajoranaLindblad(H; loss_ops=[], gain_ops=[], dephasing_ops=[], majorana_ops=[])

Deterministic BdG covariance-matrix Lindblad generator. `H` is a `BdGHamiltonian`
or a number-conserving `QuadraticHamiltonian`. The dissipators are:

- `loss_ops`  : Dirac amplitude vectors `w` (length `L`) for `L = √γ` already folded
  in, i.e. the jump operator `Σᵢ wᵢ cᵢ` (particle loss).
- `gain_ops`  : amplitude vectors for `Σᵢ wᵢ cᵢ⁺` (particle gain).
- `dephasing_ops` : `(mode_or_projector, γ)` tuples (occupation monitoring), as in
  [`CorrelationLindblad`](@ref).
- `majorana_ops` : raw 2L-component complex Majorana jump vectors `l` for a general
  linear jump `Σ_a l_a ω_a` — use these for pairing/number-non-conserving baths.
"""
function MajoranaLindblad(H::BdGHamiltonian; loss_ops=[], gain_ops=[], dephasing_ops=[],
                          majorana_ops=[])
    K = Matrix(H)
    twoL = size(K, 1)
    L = twoL ÷ 2

    B = zeros(ComplexF64, twoL, twoL)
    for w in loss_ops
        v = _dense_mode_vector(w, L)
        l = _majorana_jump_vector(v, zeros(ComplexF64, L), L)
        B .+= l * l'
    end
    for w in gain_ops
        v = _dense_mode_vector(w, L)
        l = _majorana_jump_vector(zeros(ComplexF64, L), v, L)
        B .+= l * l'
    end
    for l in majorana_ops
        length(l) == twoL ||
            throw(ArgumentError("Majorana jump vector length $(length(l)) does not match 2L = $twoL"))
        lc = Vector{ComplexF64}(l)
        all(isfinite, lc) || throw(ArgumentError("Majorana jump vector entries must be finite"))
        B .+= lc * lc'
    end

    quad = Matrix{Float64}[]
    for op in dephasing_ops
        γ, P = _dephasing_entry_majorana(op, L)
        push!(quad, sqrt(γ) .* _majoranaform_block(P))
    end

    MajoranaLindblad(K, real.(B), imag.(B), quad, nothing)
end

MajoranaLindblad(H::QuadraticHamiltonian; kwargs...) = MajoranaLindblad(BdGHamiltonian(H); kwargs...)

"""
    MajoranaLindblad(H, channels)

Build the generator from a list of `AbstractChannel`s (`Loss`, `Gain`,
`OccupationMonitor`, `HoleMonitor`), mirroring `CorrelationLindblad(H, channels)`.
"""
function MajoranaLindblad(H::Union{BdGHamiltonian,QuadraticHamiltonian}, channels)
    K = H isa BdGHamiltonian ? Matrix(H) : Matrix(BdGHamiltonian(H))
    L = size(K, 1) ÷ 2
    loss_ops = Vector{Vector{ComplexF64}}()
    gain_ops = Vector{Vector{ComplexF64}}()
    dephasing_ops = Vector{Tuple{Matrix{ComplexF64},Float64}}()
    for ch in channels
        ch isa Union{Loss,Gain,OccupationMonitor,HoleMonitor} ||
            throw(ArgumentError("unsupported channel type for MajoranaLindblad: $(typeof(ch))"))
        ch.mode.L == L ||
            throw(ArgumentError("channel mode length $(ch.mode.L) does not match system size $L"))
        γ = _channel_rate(ch)
        v = Vector{ComplexF64}(vector(ch.mode))
        if ch isa Loss
            push!(loss_ops, sqrt(γ) .* v)
        elseif ch isa Gain
            push!(gain_ops, sqrt(γ) .* v)
        else
            push!(dephasing_ops, (v * v', γ))
        end
    end
    MajoranaLindblad(BdGHamiltonian(K); loss_ops, gain_ops, dephasing_ops)
end

function _dephasing_entry_majorana(op::Tuple, L::Integer)
    length(op) == 2 || throw(ArgumentError("dephasing entries must be (mode_or_projector, gamma)"))
    raw, γ = op
    γf = Float64(γ)
    isfinite(γf) && γf ≥ 0 ||
        throw(ArgumentError("dephasing rate must be finite and non-negative, got $γf"))
    (γf, _dephasing_single_particle(raw, L))
end
_dephasing_entry_majorana(op, L::Integer) =
    throw(ArgumentError("dephasing entries must be (mode_or_projector, gamma)"))

_dephasing_single_particle(v::AbstractVector, L::Integer) =
    (m = _dense_mode_vector(v, L); m * m')
function _dephasing_single_particle(P::AbstractMatrix, L::Integer)
    size(P) == (L, L) || throw(ArgumentError("dephasing projector size $(size(P)) does not match ($L, $L)"))
    Pm = Matrix{ComplexF64}(P)
    all(isfinite, Pm) || throw(ArgumentError("dephasing projector entries must be finite"))
    Pm
end
function _dephasing_single_particle(qm::QuasiMode, L::Integer)
    qm.L == L || throw(ArgumentError("QuasiMode length $(qm.L) does not match system size $L"))
    v = _dense_mode_vector(vector(qm), L)
    v * v'
end

#---------------------------------------------------------------------------------------------------
# Right-hand side and affine generator
#---------------------------------------------------------------------------------------------------
"""
    majorana_lindblad_rhs(Liouv, Γ) -> Matrix{Float64}

The covariance time-derivative `∂ₜΓ` (real antisymmetric).
"""
function majorana_lindblad_rhs(Liouv::MajoranaLindblad, Γ::AbstractMatrix)
    G = Matrix{Float64}(Γ)
    size(G) == size(Liouv.K) ||
        throw(ArgumentError("covariance size $(size(G)) does not match generator size $(size(Liouv.K))"))
    dΓ = Liouv.K * G .- G * Liouv.K
    dΓ .-= 2 .* (Liouv.Br * G .+ G * Liouv.Br)
    dΓ .+= 4 .* Liouv.Bi
    for M in Liouv.quad
        M2 = M * M
        dΓ .+= 0.5 .* (M2 * G .+ G * M2) .- M * G * M
    end
    dΓ
end

majorana_lindblad_rhs(Liouv::MajoranaLindblad, s::MajoranaState) =
    majorana_lindblad_rhs(Liouv, covariance_matrix(s))

function _hash_matrix_real(A::AbstractMatrix, seed::UInt)
    h = hash(size(A), seed)
    for value in A
        h = hash(value, h)
    end
    h
end

function _fingerprint(Liouv::MajoranaLindblad)
    h = hash(:MajoranaLindbladGenerator, zero(UInt))
    h = _hash_matrix_real(Liouv.K, h)
    h = _hash_matrix_real(Liouv.Br, h)
    h = _hash_matrix_real(Liouv.Bi, h)
    h = hash(length(Liouv.quad), h)
    for M in Liouv.quad
        h = _hash_matrix_real(M, h)
    end
    h
end

# The RHS is affine in Γ: vec(∂ₜΓ) = M·vec(Γ) + b. Build (M, b) once and cache.
function _affine_generator(Liouv::MajoranaLindblad)
    fingerprint = _fingerprint(Liouv)
    if Liouv.cache !== nothing
        cached_fingerprint, M, b = Liouv.cache
        cached_fingerprint == fingerprint && return M, b
    end

    twoL = size(Liouv.K, 1)
    n = twoL * twoL
    Zero = zeros(Float64, twoL, twoL)
    b = vec(majorana_lindblad_rhs(Liouv, Zero))
    M = zeros(Float64, n, n)
    E = zeros(Float64, twoL, twoL)
    for k in 1:n
        E[k] = 1
        M[:, k] .= vec(majorana_lindblad_rhs(Liouv, E)) .- b
        E[k] = 0
    end
    Liouv.cache = (fingerprint, M, b)
    M, b
end

_antisymmetrize(G::AbstractMatrix) = (Gm = Matrix{Float64}(G); (Gm .- transpose(Gm)) ./ 2)

#---------------------------------------------------------------------------------------------------
# Evolution and steady state
#---------------------------------------------------------------------------------------------------
"""
    evolve!(s::MajoranaState, Liouv::MajoranaLindblad, dt) -> s

Evolve the covariance for time `dt` under the deterministic generator (dense
affine matrix exponential). Mutates and returns `s`.
"""
function evolve!(s::MajoranaState, Liouv::MajoranaLindblad, dt::Real; method::Symbol=:expm)
    method == :expm || throw(ArgumentError("unsupported MajoranaLindblad evolution method: $method"))
    size(s.Gamma) == size(Liouv.K) ||
        throw(ArgumentError("MajoranaState has covariance size $(size(s.Gamma)) but generator is $(size(Liouv.K))"))
    M, b = _affine_generator(Liouv)
    n = length(b)
    A = zeros(Float64, n + 1, n + 1)
    A[1:n, 1:n] .= M
    A[1:n, n + 1] .= b
    y = exp(dt .* A) * vcat(vec(s.Gamma), 1.0)
    s.Gamma = _antisymmetrize(reshape(y[1:n], size(Liouv.K)))
    s
end

evolve(s::MajoranaState, Liouv::MajoranaLindblad, dt::Real; kwargs...) =
    evolve!(copy(s), Liouv, dt; kwargs...)

"""
    steadystate(Liouv::MajoranaLindblad; atol=1e-10, rtol=1e-8) -> MajoranaState

Solve `∂ₜΓ = 0` for a unique steady covariance via the dense affine generator.
Errors if the generator is numerically singular (non-unique steady state) or the
solution is unphysical (eigenvalues of `iΓ` outside `[-1, 1]`).
"""
function steadystate(Liouv::MajoranaLindblad; atol::Real=1e-10, rtol::Real=1e-8,
                     rank_rtol::Real=eps(Float64), check_unique::Bool=true)
    M, b = _affine_generator(Liouv)
    if check_unique
        svals = svdvals(M)
        threshold = rank_rtol * max(size(M)...) * maximum(svals)
        minimum(svals) ≤ threshold &&
            throw(ArgumentError("MajoranaLindblad steady state is not unique; the dense generator is numerically singular"))
    end

    x = try
        -(M \ b)
    catch err
        throw(ArgumentError("MajoranaLindblad steady-state solve failed: $(err)"))
    end

    twoL = size(Liouv.K, 1)
    G = _antisymmetrize(reshape(x, twoL, twoL))
    residual = norm(majorana_lindblad_rhs(Liouv, G))
    residual ≤ atol + rtol * (norm(M) * norm(x) + norm(b)) ||
        throw(ArgumentError("MajoranaLindblad steady-state residual $residual exceeds tolerance"))

    MajoranaState(G; check=true, atol=sqrt(atol))
end
