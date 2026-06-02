#---------------------------------------------------------------------------------------------------
# Evolution, gates, and the quantum-trajectory engine
#---------------------------------------------------------------------------------------------------
export evolve, evolve!, Gate, apply!
export step!, step_diffusive!, measure!, Feedback
export NonHermitianGenerator, effective_hamiltonian, noclick_propagator, evolve_noclick!
export ensemble, EnsembleResult

#---------------------------------------------------------------------------------------------------
# Re-orthonormalization (pure states only)
#---------------------------------------------------------------------------------------------------
"""    normalize!(s::SlaterState) — re-orthonormalize the orbitals via QR."""
function LinearAlgebra.normalize!(s::SlaterState)
    s.B = Matrix(qr!(s.B).Q)
    s
end

#---------------------------------------------------------------------------------------------------
# Whole-system unitary evolution
#---------------------------------------------------------------------------------------------------
"""
    evolve!(s, U)            apply a single-particle unitary `U`
    evolve!(s, H, dt)        evolve for time `dt` under a `QuadraticHamiltonian`
    evolve(s, ...)           non-mutating copy

Pure (`SlaterState`): `B ← U B`. Mixed (`CorrelationState`): `C ← U C U'`.
"""
function evolve!(s::SlaterState, U::AbstractMatrix)
    s.B = U * s.B
    s
end
function evolve!(s::CorrelationState, U::AbstractMatrix)
    # orbitals evolve B → U B, so C = conj(B)Bᵀ transforms as C → Ū C Uᵀ
    s.C = Hermitian(conj(U) * s.C * transpose(U))
    s
end
evolve!(s::NumberConservingGaussianState, H::QuadraticHamiltonian, dt::Real) = evolve!(s, propagator(H, dt))
evolve(s::NumberConservingGaussianState, args...) = evolve!(copy(s), args...)

#---------------------------------------------------------------------------------------------------
# Local gates
#---------------------------------------------------------------------------------------------------
"""    Gate(M, I) — a local single-particle gate `M` acting on sites `I`."""
struct Gate{T<:AbstractMatrix}
    M::T
    I::Vector{Int64}
end

"""
    apply!(M, s::SlaterState, inds; normalize=false, threads=false)

Apply a local single-particle matrix `M` on sites `inds` (in place, turbo kernels).
"""
function apply!(M::AbstractMatrix, s::SlaterState, inds::AbstractVector{<:Integer};
               normalize::Bool=false, threads::Bool=false)
    B = s.B[inds, :]
    Bv = view(s.B, inds, :)
    threads ? tturbo_mul!(Bv, M, B) : turbo_mul!(Bv, M, B)
    normalize && normalize!(s)
    s
end
function apply!(M::AbstractMatrix, s::CorrelationState, inds::AbstractVector{<:Integer}; kwargs...)
    # consistent with B → M B on the support: C → conj(M) C transpose(M) on `inds`
    C = Matrix(s.C)
    C[inds, :] = conj(M) * C[inds, :]
    C[:, inds] = C[:, inds] * transpose(M)
    s.C = Hermitian(C)
    s
end
apply!(g::Gate, s::NumberConservingGaussianState; kwargs...) = apply!(g.M, s, g.I; kwargs...)

function apply!(gates::AbstractVector{<:Gate}, s::NumberConservingGaussianState;
               normalize::Bool=false, threads::Bool=length(gates) > 100)
    if threads
        Threads.@threads for g in gates
            apply!(g, s)
        end
    else
        for g in gates
            apply!(g, s)
        end
    end
    normalize && normalize!(s)
    s
end

#---------------------------------------------------------------------------------------------------
# One Monte-Carlo wave-function (quantum-jump) step
#---------------------------------------------------------------------------------------------------
"""
    step!(s::SlaterState, U, channels, dt; rng)
    step!(s::SlaterState, H::QuadraticHamiltonian, channels, dt; rng)

One MCWF step: apply the deterministic unitary, then for each channel draw a
click (apply the jump) or no-click (apply the back-action). The state is kept
orthonormal.
"""
function step!(s::SlaterState, U::AbstractMatrix, channels, dt::Real; rng::AbstractRNG=Random.default_rng())
    evolve!(s, U)
    normQ = true
    for ch in channels
        rate, v = jump_rate(ch, s, dt)
        if rand(rng) < rate
            apply_click!(ch, s, v)
            normalize!(s)
            normQ = true
        else
            apply_noclick!(ch, s, dt; normalize=false, threads=true)
            normQ = false
        end
    end
    normQ || normalize!(s)
    s
end
step!(s::SlaterState, H::QuadraticHamiltonian, channels, dt::Real; rng::AbstractRNG=Random.default_rng()) =
    step!(s, propagator(H, dt), channels, dt; rng)

#---------------------------------------------------------------------------------------------------
# No-click / non-Hermitian deterministic evolution (post-selected trajectory)
#---------------------------------------------------------------------------------------------------
"""    NonHermitianGenerator(Heff) — wraps the effective single-particle Hamiltonian."""
struct NonHermitianGenerator{T}
    Heff::Matrix{T}
end

"""
    effective_hamiltonian(H::QuadraticHamiltonian, channels) -> NonHermitianGenerator

`H_eff = h - (i/2) Σ L†L` (single-particle), the generator of the no-click branch.
"""
function effective_hamiltonian(H::QuadraticHamiltonian, channels)
    L = nmodes(H)
    Heff = Matrix{ComplexF64}(Matrix(H.h))
    for ch in channels
        Heff .-= (im / 2) .* single_particle_LdagL(ch, L)
    end
    NonHermitianGenerator(Heff)
end

"""    noclick_propagator(G, dt) = exp(-i H_eff dt)  (non-unitary)."""
noclick_propagator(G::NonHermitianGenerator, dt::Real) = exp(-im * dt * G.Heff)

"""
    evolve_noclick!(s::SlaterState, G, dt; renormalize=true) -> w

Deterministic no-click evolution `B ← exp(-i H_eff dt) B`, returning the no-click
weight `w = ‖unnormalized‖²` of the step (accumulate `log(w)` along a trajectory).
"""
function evolve_noclick!(s::SlaterState, G::NonHermitianGenerator, dt::Real; renormalize::Bool=true)
    Bnew = noclick_propagator(G, dt) * s.B
    w = real(det(Bnew' * Bnew))
    s.B = Bnew
    renormalize && normalize!(s)
    w
end

#---------------------------------------------------------------------------------------------------
# Continuous (diffusive / quantum-state-diffusion) monitoring
#---------------------------------------------------------------------------------------------------
"""
    step_diffusive!(s::SlaterState, channels, dt; rng)

One Wiener (QSD) step of continuous occupation monitoring:
`ψ → exp{∑ⱼ[δWⱼ + (2⟨nⱼ⟩-1)γ dt] nⱼ} ψ`. Channels must be `OccupationMonitor`.
"""
function step_diffusive!(s::SlaterState, channels, dt::Real; rng::AbstractRNG=Random.default_rng())
    for ch in channels
        ch isa OccupationMonitor || error("step_diffusive! supports OccupationMonitor channels only")
        qm = ch.mode
        p = inner(qm, s.B)
        a = randn(rng) * sqrt(ch.γ * dt) + (2 * real(dot(p, p)) - 1) * ch.γ * dt
        m = (exp(a) - 1) * qm.V * qm.V' + I
        apply!(m, s, qm.I)
    end
    normalize!(s)
    s
end

#---------------------------------------------------------------------------------------------------
# Projective measurement
#---------------------------------------------------------------------------------------------------
"""
    measure!(qm::QuasiMode, s::SlaterState; rng) -> Bool

Projective occupation measurement of mode `qm` with Born statistics; returns the
outcome (`true` = occupied).
"""
function measure!(qm::QuasiMode, s::SlaterState; rng::AbstractRNG=Random.default_rng())
    p = inner(qm, s.B)
    if rand(rng) < real(dot(p, p))
        replace_vector!(s.B, vector(qm), p); normalize!(s); return true
    else
        avoid_vector!(s.B, vector(qm), p); normalize!(s); return false
    end
end

#---------------------------------------------------------------------------------------------------
# Measurement-conditioned feedback
#---------------------------------------------------------------------------------------------------
"""
    Feedback(channel, on_click, on_noclick)

A channel whose click / no-click outcome triggers a feedback unitary
(`on_click` / `on_noclick`) on the channel's support.
"""
struct Feedback{C<:AbstractChannel,U<:AbstractMatrix}
    channel::C
    on_click::U
    on_noclick::U
end

"""    apply!(fb::Feedback, s::SlaterState, dt; rng) -> Bool (whether a click occurred)."""
function apply!(fb::Feedback, s::SlaterState, dt::Real; rng::AbstractRNG=Random.default_rng())
    ch = fb.channel
    rate, v = jump_rate(ch, s, dt)
    if rand(rng) < rate
        apply_click!(ch, s, v); normalize!(s)
        apply!(fb.on_click, s, ch.mode.I; threads=true)
        return true
    else
        apply_noclick!(ch, s, dt; normalize=true)
        apply!(fb.on_noclick, s, ch.mode.I; threads=true)
        return false
    end
end

#---------------------------------------------------------------------------------------------------
# Ensemble / trajectory-average runner
#---------------------------------------------------------------------------------------------------
"""
    EnsembleResult

Result of [`ensemble`](@ref): `saveat` (times), `mean` and `std` (NamedTuples of
per-observable time series), and `ntraj`.
"""
struct EnsembleResult
    saveat::Vector{Float64}
    mean::NamedTuple
    std::NamedTuple
    ntraj::Int
end

"""
    ensemble(init, H, channels; ntraj, tspan, dt, observables,
             rng=Random.default_rng(), parallel=:threads, postselect=false) -> EnsembleResult

Run `ntraj` quantum-trajectory simulations and average each observable on a time
grid. `init` is a zero-arg function returning a fresh `SlaterState`. `H` is a
`QuadraticHamiltonian` (or a unitary matrix when `postselect=false`). `observables`
is a `NamedTuple` of functions `state -> value` (scalar or array). With
`postselect=true` the deterministic no-click branch is used.
"""
function ensemble(init, H, channels; ntraj::Integer, tspan::Real, dt::Real,
                  observables::NamedTuple, rng::AbstractRNG=Random.default_rng(),
                  parallel::Symbol=:threads, postselect::Bool=false)
    nsteps = round(Int, tspan / dt)
    saveat = collect(dt .* (1:nsteps))
    obsfns = values(observables)
    nobs = length(obsfns)
    Upre = (H isa QuadraticHamiltonian && !postselect) ? propagator(H, dt) : H
    G = postselect ? effective_hamiltonian(H, channels) : nothing
    seeds = rand(rng, UInt64, ntraj)

    # one trajectory -> Vector (per observable) of length-nsteps Vector{Any}
    runone = function (seed)
        r = Random.Xoshiro(seed)
        s = init()
        ser = [Vector{Any}(undef, nsteps) for _ in 1:nobs]
        for k in 1:nsteps
            postselect ? evolve_noclick!(s, G, dt) : step!(s, Upre, channels, dt; rng=r)
            for j in 1:nobs
                ser[j][k] = obsfns[j](s)
            end
        end
        ser
    end

    # accumulate Σx and Σx² over a set of trajectory indices
    accumulate = function (idxs)
        sum = nothing; sumsq = nothing
        for i in idxs
            ser = runone(seeds[i])
            if sum === nothing
                sum   = [Vector{Any}([ser[j][k]      for k in 1:nsteps]) for j in 1:nobs]
                sumsq = [Vector{Any}([abs2.(ser[j][k]) for k in 1:nsteps]) for j in 1:nobs]
            else
                for j in 1:nobs, k in 1:nsteps
                    sum[j][k]   = sum[j][k]   .+ ser[j][k]
                    sumsq[j][k] = sumsq[j][k] .+ abs2.(ser[j][k])
                end
            end
        end
        (sum, sumsq)
    end

    nchunks = parallel == :threads ? max(1, min(Threads.nthreads(), ntraj)) : 1
    chunks = dividerange(collect(1:ntraj), nchunks)
    partials = Vector{Any}(undef, nchunks)
    if parallel == :threads
        Threads.@threads for c in 1:nchunks
            partials[c] = accumulate(chunks[c])
        end
    else
        for c in 1:nchunks
            partials[c] = accumulate(chunks[c])
        end
    end

    # combine partial sums
    tot   = [Vector{Any}(undef, nsteps) for _ in 1:nobs]
    totsq = [Vector{Any}(undef, nsteps) for _ in 1:nobs]
    for j in 1:nobs, k in 1:nsteps
        tot[j][k]   = mapreduce(p -> p[1][j][k], (a, b) -> a .+ b, partials)
        totsq[j][k] = mapreduce(p -> p[2][j][k], (a, b) -> a .+ b, partials)
    end

    meanvals = map(1:nobs) do j
        [tot[j][k] ./ ntraj for k in 1:nsteps]
    end
    stdvals = map(1:nobs) do j
        [sqrt.(max.(totsq[j][k] ./ ntraj .- abs2.(tot[j][k] ./ ntraj), 0.0)) for k in 1:nsteps]
    end

    ks = keys(observables)
    EnsembleResult(saveat, NamedTuple{ks}(Tuple(meanvals)), NamedTuple{ks}(Tuple(stdvals)), Int(ntraj))
end

#---------------------------------------------------------------------------------------------------
# Helper: split a vector into `nthreads` contiguous chunks
#---------------------------------------------------------------------------------------------------
function dividerange(vec::AbstractVector, nthreads::Integer)
    list = Vector{Vector{eltype(vec)}}(undef, nthreads)
    eachthreads, left = divrem(length(vec), nthreads)
    start = 1
    for i = 1:left
        stop = start + eachthreads
        list[i] = vec[start:stop]
        start = stop + 1
    end
    for i = left+1:nthreads-1
        stop = start + eachthreads - 1
        list[i] = vec[start:stop]
        start = stop + 1
    end
    list[nthreads] = vec[start:end]
    list
end
