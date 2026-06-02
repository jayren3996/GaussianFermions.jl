include("LinAlg.jl")
#----------------------------------------------------------------------------------------------------
# Free Fermion States
#----------------------------------------------------------------------------------------------------
export FreeFermionState
"""
Particle-number-preserving free fermion state

Represented by a matrix `B`,
    |ψ⟩ = ⨂ₖ bₖ⁺|0⟩
    bₖ⁺ = ∑ᵢ cᵢ⁺ Bᵢₖ.
A `FreeFermionState` object `s` can be access as a matrix.
The element `s[i,j]` is the two-point function ⟨cᵢ⁺cⱼ⟩.

Notes:
------
Since most of the modifications of the free fermion state is designed to be inplace, by default the
element type of B should be complex.
"""
mutable struct FreeFermionState{T<:Number}
    B::Matrix{T}
end

#----------------------------------------------------------------------------------------------------
# Construction
#----------------------------------------------------------------------------------------------------
"""
    FreeFermionState(pos::AbstractVector{<:Integer}, L::Integer; dtype::DataType)

Initialize FreeFermionState with given data type, particle positions and system length.

Inputs:
-------
pos  : vector of occupied positions.
L    : length.
dtype: data type of representing matrix B, default to be complex.
"""
function FreeFermionState(pos::AbstractVector{<:Integer}, L::Integer; dtype::DataType=ComplexF64)
    B = zeros(dtype, L, length(pos))
    for (i, ind) in enumerate(pos)
        B[ind, i] = one(dtype)
    end
    FreeFermionState(B)
end
#----------------------------------------------------------------------------------------------------
"""
    FreeFermionState(dtype::DataType, vec::AbstractVector{Bool})

Initialize FreeFermionState with vector indicating particle positions.

Inputs:
-------
pos  : vector of Boolean values indicating whether a site is occupied.
dtype: data type of representing matrix B.
"""
function FreeFermionState(vec::AbstractVector{Bool}; dtype::DataType=ComplexF64)
    pos = Int64[]
    for (i, b) in enumerate(vec)
        b && push!(pos, i)
    end
    FreeFermionState(pos, length(vec); dtype)
end
#----------------------------------------------------------------------------------------------------
"""
    FreeFermionState(;L::Integer, N::Integer, config::String="Z2")

More initial configuration.

Inputs:
-------
L     : Length.
N     : Number of fermions.
config: Product state configuration:
    - left/right/center
    - random 
    - Zn
"""
function FreeFermionState(;L::Integer, N::Integer, config::String="Z2")
    pos = if config == "center"
        l = (L-N)÷2
        l+1:l+N
    elseif config == "left"
        1:N
    elseif config == "right"
        L-N+1:L
    elseif config == "random"
        shuffle(1:L)[1:N]
    elseif config[1] == 'Z'
        n = parse(Int, config[2])
        1:n:1+n*(N-1)
    else
        error("Invalid configureation: $config.")
    end
    FreeFermionState(pos, L)
end


#----------------------------------------------------------------------------------------------------
# Properties
#----------------------------------------------------------------------------------------------------
Base.eltype(::FreeFermionState{T}) where {T} = T
Base.length(s::FreeFermionState) = size(s.B, 1)
#----------------------------------------------------------------------------------------------------
"""
Compute correlation ⟨cᵢ⁺cⱼ⟩
"""
function Base.getindex(s::FreeFermionState, i::Integer, j::Integer) 
    Bi = view(s.B, i, :)
    Bj = view(s.B, j, :)
    return dot(Bi, Bj)
end
#----------------------------------------------------------------------------------------------------
"""
Compute correlation matrix
"""
function Base.getindex(s::FreeFermionState, ::Colon) 
    return conj(s.B) * transpose(s.B)
end
function Base.getindex(s::FreeFermionState, ::Colon, ::Colon) 
    return conj(s.B) * transpose(s.B)
end
#----------------------------------------------------------------------------------------------------
"""
Compute particle density on each site.
"""
function LinearAlgebra.diag(s::FreeFermionState)
    n = Vector{Float64}(undef,length(s))
    for i in 1:length(s)
        n[i] = real(s[i,i])
    end
    n
end
#----------------------------------------------------------------------------------------------------
"""
Return particle number.
"""
function LinearAlgebra.rank(s::FreeFermionState) 
    return size(s.B, 2)
end
#----------------------------------------------------------------------------------------------------
export ent_S
"""
    entropy(s::FreeFermionState, i::AbstractVector{<:Integer})

Entaglement entropy of gaussian state with system A chosen to be the sites {i}.
"""
function ent_S(s::FreeFermionState, i::AbstractVector{<:Integer})
    vals = svdvals(s.B[i,:])
    EE = 0.0
    for val in vals
        val < 1e-7 && continue
        val > 1.0 && val-1.0 > 1e-6 && error("Got a Schmidt value λ = $val.")
        x = val ^ 2
        (y = 1.0 - x) < 1e-14 && continue
        EE -= x * log(x) + y * log(y)
    end
    EE
end
#----------------------------------------------------------------------------------------------------
"""
    entropy(s::FreeFermionState, i::AbstractVector{<:Integer}, j::AbstractVector{<:Integer})

Mutual information:
    I(A:B) = S(A) + S(B) - S(A∪B)
"""
function ent_S(
    s::FreeFermionState, 
    i::AbstractVector{<:Integer}, 
    j::AbstractVector{<:Integer}
)
    SA = ent_S(s, i)
    SB = ent_S(s, j)
    SAB = ent_S(s, vcat(i,j))
    SA + SB - SAB
end


#----------------------------------------------------------------------------------------------------
# Operations
#----------------------------------------------------------------------------------------------------
"""
Normalize FreeFermionState.
"""
function LinearAlgebra.normalize!(s::FreeFermionState) 
    s.B = Matrix(qr!(s.B).Q)
end



#----------------------------------------------------------------------------------------------------
# Gate Operation
#----------------------------------------------------------------------------------------------------
export Gate
struct Gate{T<:AbstractMatrix}
    M::T
    I::Vector{Int64}
end
#----------------------------------------------------------------------------------------------------
function *(M::AbstractMatrix, s::FreeFermionState)
    B = M * s.B
    FreeFermionState(B)
end
#----------------------------------------------------------------------------------------------------
export apply!
"""
Multiply general matrix to FreeFermionState, and by default normalize the output.
"""
function apply!(M::AbstractMatrix, s::FreeFermionState; normalize::Bool=false)
    s.B = M * s.B
    normalize && normalize!(s)
    return s
end
#----------------------------------------------------------------------------------------------------
"""
    apply!(U, s::FreeFermionState, ind::AbstractVector{<:Integer})

Apply local unitary to FreeFermionState `s` on sites `inds`.
"""
function apply!(
    M::AbstractMatrix, 
    s::FreeFermionState, 
    inds::AbstractVector{<:Integer}; 
    normalize::Bool=false,
    threads::Bool=false
)
    B = s.B[inds, :]
    Bv = view(s.B, inds, :)
    threads ? tturbo_mul!(Bv, M, B) : turbo_mul!(Bv, M, B)
    normalize && normalize!(s)
    return s
end
function apply!(
    G::Gate, s::FreeFermionState; 
    normalize::Bool=false,
    threads::Bool=false
) 
    apply!(G.M, s, G.I; normalize, threads)
end
#----------------------------------------------------------------------------------------------------
"""
    apply!(U, s::FreeFermionState, ind::AbstractVector{<:Integer})

Apply local unitary gate to FreeFermionState `s` on sites `inds`.
"""
function apply!(
    gates::AbstractVector{<:Gate}, 
    s::FreeFermionState;
    normalize::Bool=false,
    threads::Bool=length(gates)>100
)
    if threads
        Threads.@threads for gate in gates
            apply!(gate, s)
        end
    else
        for gate in gates
            apply!(gate, s)
        end
    end
    normalize && normalize!(s)
    return s
end

#----------------------------------------------------------------------------------------------------
# Quantum Jumps
#----------------------------------------------------------------------------------------------------
export QuasiMode
"""
Quasi Mode d⁺ = ∑ Vⱼ cⱼ⁺
"""
struct QuasiMode{T<:Number}
    I::Vector{Int64}
    V::Vector{T}
    L::Int64
    function QuasiMode(I::AbstractVector{<:Integer}, V::AbstractVector{T}, L::Integer) where T <: Number 
        ind = [mod(i-1, L)+1 for i in I]
        new{T}(ind, V, Int64(L))
    end
end
#----------------------------------------------------------------------------------------------------
vector(qm::QuasiMode) = sparsevec(qm.I, qm.V, qm.L)
#----------------------------------------------------------------------------------------------------
function inner!(out::Vector, qm::QuasiMode, s::FreeFermionState) 
    turbo_dot!(out, qm.V, s.B, qm.I)
end
function inner(qm::QuasiMode, s::FreeFermionState) 
    out = Vector{ComplexF64}(undef, size(s.B, 2))
    inner!(out, qm, s)
    out
end
#----------------------------------------------------------------------------------------------------
export QuantumJump
"""
abstract type for quantum jumps that preserve free fermion structure.
"""
abstract type QuantumJump end
(qj::QuantumJump)(I, V, L, γdt) = qj(QuasiMode(I, V, L), γdt)
#----------------------------------------------------------------------------------------------------
export QJParticle
"""
QJParticle: L = d⁺d
"""
struct QJParticle{T1, T2<:AbstractMatrix, T3<:Real} <: QuantumJump
    M::QuasiMode{T1}
    P::T2
    γdt::T3
end
QJParticle(M::QuasiMode, γdt::Real) = QJParticle(M, particle_exp(M, γdt), γdt)
#----------------------------------------------------------------------------------------------------
export QJHole
"""
QJHole: L = dd⁺
"""
struct QJHole{T1, T2<:AbstractMatrix, T3<:Real} <: QuantumJump
    M::QuasiMode{T1}
    P::T2
    γdt::T3
end
QJHole(M::QuasiMode, γdt::Real) = QJHole(M, hole_exp(M, γdt), γdt)
#----------------------------------------------------------------------------------------------------
export QJDrain
"""
QJDrain: L = d 
"""
struct QJDrain{T1, T2<:AbstractMatrix, T3<:Real} <: QuantumJump
    M::QuasiMode{T1}
    P::T2
    γdt::T3
end
QJDrain(M::QuasiMode, γdt::Real) = QJDrain(M, particle_exp(M, γdt), γdt)
#----------------------------------------------------------------------------------------------------
export QJSource
"""
QJSource: L = d⁺
"""
struct QJSource{T1, T2<:AbstractMatrix, T3<:Real} <: QuantumJump
    M::QuasiMode{T1}
    P::T2
    γdt::T3
end
QJSource(M::QuasiMode, γdt::Real) = QJSource(M, hole_exp(M, γdt), γdt)
#----------------------------------------------------------------------------------------------------
"""
Correaction operator if jump does not happen
    P = sqrt(1-γ⋅dt)d⁺d + dd⁺
"""
function particle_exp(qm::QuasiMode, γdt::Real)
    v = qm.V
    n = length(v)
    mat = (sqrt(1-γdt)-1) * v * v' + I(n)
    SMatrix{n,n}(mat)
end
#----------------------------------------------------------------------------------------------------
"""
Correaction operator if jump does not happen
    P = d⁺d + sqrt(1-γ⋅dt)dd⁺
"""
function hole_exp(qm::QuasiMode, γdt::Real)
    v = qm.V
    n = length(v)
    a = sqrt(1-γdt)
    mat = (1-a) * v * v' + a * I(n)
    SMatrix{n,n}(mat)
end


#----------------------------------------------------------------------------------------------------
# Weak measurement
#----------------------------------------------------------------------------------------------------
function apply!(qj::QJParticle, s::FreeFermionState, p::AbstractVector; renorm::Bool=true)    # L = d⁺d
    replace_vector!(s.B, vector(qj.M), p)
    renorm && normalize!(s)
    return s
end
function apply!(qj::QJHole, s::FreeFermionState, p::AbstractVector; renorm::Bool=true)        # L = dd⁺
    avoid_vector!(s.B, vector(qj.M), p)
    renorm && normalize!(s)
    return s
end
function apply!(::QJDrain, s::FreeFermionState, p::AbstractVector; renorm::Bool=true)         # L = d
    s.B = delete_vector(s.B, p)
    renorm && normalize!(s)
    return s
end
function apply!(qj::QJSource, s::FreeFermionState, ::AbstractVector; renorm::Bool=true)       # L = d⁺
    s.B = insert_vector(s.B, vector(qj.M))
    renorm && normalize!(s)
    return s
end
#----------------------------------------------------------------------------------------------------
"""
Apply quantum jump on a free fermion state `s`

Return whether jump happened
"""
function apply!(qj::QuantumJump, s::FreeFermionState)
    v = inner(qj.M, s)
    if rand() < real(dot(v, v)) * qj.γdt
        apply!(qj, s, v)
        return true
    else
        apply!(qj.P, s, qj.M.I; normalize=true, threads=true)
        return false
    end
end
#----------------------------------------------------------------------------------------------------
"""
Apply multiple quantum jumps on a free fermion state.

Note that indices of the jumps should have no overlap.
"""
function apply!(qjs::AbstractVector{<:QuantumJump}, s::FreeFermionState)
    normQ = true
    for qj in qjs
        # allocate per jump: QJDrain/QJSource change the number of columns of B
        v = inner(qj.M, s)
        if rand() < real(dot(v, v)) * qj.γdt
            apply!(qj, s, v)
            normQ = true
        else
            apply!(qj.P, s, qj.M.I; threads=true)
            normQ = false
        end
    end
    normQ || normalize!(s)
    return s
end

#----------------------------------------------------------------------------------------------------
# Conditional Jump
#----------------------------------------------------------------------------------------------------
export ConditionalJump
"""
Quantum jump with conditional feedback 

First decided if jump happens (true/false), then apply (Ut/Uf) feedback.
"""
struct ConditionalJump{T<:QuantumJump, Tm<:AbstractMatrix}
    J::T
    U::Tm
end
#----------------------------------------------------------------------------------------------------
function apply!(cj::ConditionalJump, s::FreeFermionState)
    v = inner(cj.J.M, s)
    ind = cj.J.M.I
    if rand() < real(dot(v, v)) * cj.J.γdt
        apply!(cj.J, s, v)
        apply!(cj.U, s, ind)
        return true
    else
        apply!(cj.J.P, s, ind; normalize=true, threads=true)
        return false
    end
end
#----------------------------------------------------------------------------------------------------
"""
Apply a list of quantum jumps with condional feedback.

Note that indices of the jumps should have no overlap.
"""
function apply!(cjs::AbstractVector{<:ConditionalJump}, s::FreeFermionState)
    normQ = true
    for cj in cjs
        qj = cj.J
        ind = qj.M.I
        # allocate per jump: QJDrain/QJSource change the number of columns of B
        v = inner(qj.M, s)
        if rand() < real(dot(v, v)) * qj.γdt
            apply!(qj, s, v)
            apply!(cj.U, s, ind)
            normQ = true
        else
            apply!(qj.P, s, ind; threads=true)
            normQ = false
        end
    end
    normQ || normalize!(s)
    return s
end


#----------------------------------------------------------------------------------------------------
# Gaussian SSE
#----------------------------------------------------------------------------------------------------
export wiener!
"""
Wiener process:
    ψ → exp{∑ⱼ[δWⱼ + (2⟨nⱼ⟩-1)γ δt]nⱼ} ψ
"""
function wiener!(
    qms::AbstractVector{<:QuasiMode}, 
    s::FreeFermionState, 
    γdt::Real; 
    threads::Bool=length(qms)>100
)
    sgdt = sqrt(γdt)
    if threads
        Threads.@threads for qmsi in dividerange(qms, Threads.nthreads())
            _wiener_for!(qmsi, s, γdt, sgdt)
        end
    else
        _wiener_for!(qms, s, γdt, sgdt)
    end
    normalize!(s)
end

function _wiener_for!(
    qms::AbstractVector{<:QuasiMode}, 
    s::FreeFermionState, 
    γdt::Real, sgdt::Real
)
    p = Vector{ComplexF64}(undef, size(s.B, 2))
    for qm in qms
        inner!(p, qm, s)
        a = randn() * sgdt + (2 * real(dot(p, p)) - 1) * γdt
        m = (exp(a) - 1) * qm.V * qm.V' + I
        apply!(m, s, qm.I)
    end
end


#----------------------------------------------------------------------------------------------------
# Projective measurement
#----------------------------------------------------------------------------------------------------
export measure!
"""
Projective Measure
"""
function measure!(qm::QuasiMode, s::FreeFermionState)
    p = inner(qm, s)
    if rand() < real(dot(p, p))
        replace_vector!(s.B, vector(qm), p) 
        normalize!(s)
        return true
    else
        avoid_vector!(s.B, vector(qm), p)
        normalize!(s)
        return false
    end
end
#----------------------------------------------------------------------------------------------------
export ConditionalMeasure
"""
Measurement with conditional feddback

First decided the outcome of the measurement (true/false), then apply (Ut/Uf) feedback.
"""
struct ConditionalMeasure{T<:QuasiMode, Tm1<:AbstractMatrix, Tm2<:AbstractMatrix}
    M::T
    Ut::Tm1
    Uf::Tm2
end
#----------------------------------------------------------------------------------------------------
function apply!(cm::ConditionalMeasure, s::FreeFermionState)
    q = measure!(cm.M, s)
    apply!(q ? cm.Ut : cm.Uf, s, cm.M.I; threads=true)
    q
end


#----------------------------------------------------------------------------------------------------
# Helper
#----------------------------------------------------------------------------------------------------
function dividerange(vec::AbstractVector, nthreads::Integer)
    list = Vector{Vector{eltype(vec)}}(undef, nthreads)
    eachthreads, left = divrem(length(vec), nthreads)
    start = 1
    for i = 1:left
        stop  = start + eachthreads
        list[i] = vec[start:stop]
        start = stop+1
    end
    for i = left+1:nthreads-1
        stop  = start + eachthreads - 1
        list[i] = vec[start:stop]
        start = stop+1
    end
    list[nthreads] = vec[start:end]
    list
end




