# Stage 1b CorrelationLindblad Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic number-conserving Lindblad evolution and steady-state solving for `CorrelationState`.

**Architecture:** Implement a new focused `src/CorrelationLindblad.jl` layer that depends on the Stage 1a state, Hamiltonian, and channel APIs, but not on `Quadratic.jl`. The Lindbladian exposes low-level linear bath constructors first, then channel adapters. Dense finite-system evolution and steady states use a cached affine vectorized generator built from the public RHS, keeping convention handling centralized and testable.

**Tech Stack:** Julia, LinearAlgebra, GaussianFermions.jl local module, existing `QuadraticHamiltonian`, `CorrelationState`, and channel types.

---

## File Structure

- Create `src/CorrelationLindblad.jl`: owns `CorrelationLindblad`, low-level bath parsing, channel adapters, `lindblad_rhs`, dense affine generator caching, `evolve!`, `evolve`, and `steadystate`.
- Modify `src/GaussianFermions.jl`: include `CorrelationLindblad.jl` after `Trajectory.jl` and before `Quadratic.jl`.
- Modify `test/runtests.jl`: add Stage 1b tests in a new `@testset "CorrelationLindblad"` near the existing Hamiltonian/evolution tests.
- Modify `README.md`: add a compact deterministic mixed-state example.

---

### Task 1: Low-Level Type, Constructors, and RHS

**Files:**
- Create: `src/CorrelationLindblad.jl`
- Modify: `src/GaussianFermions.jl`
- Test: `test/runtests.jl`

- [ ] **Step 1: Write failing RHS and constructor tests**

Add this `@testset` after the existing `"Hamiltonian & evolution"` testset in `test/runtests.jl`:

```julia
    @testset "CorrelationLindblad" begin
        unitmode(L, i) = (v = zeros(ComplexF64, L); v[i] = 1; v)

        L = 3
        H = hopping(L; pbc=false)
        C0 = ComplexF64[
            0.6   0.2im 0.0
           -0.2im 0.4   0.1
            0.0   0.1   0.3
        ]
        C0 = (C0 + C0') / 2

        lind_h = CorrelationLindblad(H)
        @test lindblad_rhs(lind_h, C0) ≈ im * conj(Matrix(H.h)) * C0 - im * C0 * transpose(Matrix(H.h))

        loss_lind = CorrelationLindblad(zeros(ComplexF64, 1, 1); loss_ops=[sqrt(0.7) * unitmode(1, 1)])
        @test lindblad_rhs(loss_lind, ComplexF64[1;;]) ≈ ComplexF64[-0.7;;]
        @test lindblad_rhs(loss_lind, CorrelationState([1.0])) ≈ ComplexF64[-0.7;;]

        gain_lind = CorrelationLindblad(zeros(ComplexF64, 1, 1); gain_ops=[sqrt(0.4) * unitmode(1, 1)])
        @test lindblad_rhs(gain_lind, ComplexF64[0;;]) ≈ ComplexF64[0.4;;]
        @test lindblad_rhs(gain_lind, ComplexF64[1;;]) ≈ ComplexF64[0;;] atol = 1e-12

        both_lind = CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                        loss_ops=[sqrt(0.6) * unitmode(1, 1)],
                                        gain_ops=[sqrt(0.2) * unitmode(1, 1)])
        @test lindblad_rhs(both_lind, ComplexF64[0.25;;]) ≈ ComplexF64[0;;] atol = 1e-12

        deph_lind = CorrelationLindblad(zeros(ComplexF64, 2, 2);
                                        dephasing_ops=[(unitmode(2, 1), 0.5)])
        Ccoh = ComplexF64[0.5 0.25; 0.25 0.5]
        dC = lindblad_rhs(deph_lind, Ccoh)
        @test diag(dC) ≈ [0, 0] atol = 1e-12
        @test dC[1, 2] ≈ -0.25 * 0.5 / 2 atol = 1e-12
        @test dC[2, 1] ≈ -0.25 * 0.5 / 2 atol = 1e-12

        v = ComplexF64[1, im] / sqrt(2)
        complex_deph = CorrelationLindblad(zeros(ComplexF64, 2, 2);
                                           dephasing_ops=[(v, 0.3)])
        Q = conj(v * v')
        @test complex_deph.dephasing[1][2] ≈ Q
    end
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL with `UndefVarError: CorrelationLindblad not defined`.

- [ ] **Step 3: Add `src/CorrelationLindblad.jl` with constructors and RHS**

Create `src/CorrelationLindblad.jl`:

```julia
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
```

- [ ] **Step 4: Include the new file**

Modify `src/GaussianFermions.jl` so the include block is:

```julia
include("LinAlg.jl")        # numerical kernels (unchanged)
include("Modes.jl")         # QuasiMode + overlaps
include("States.jl")        # SlaterState / CorrelationState
include("Observables.jl")   # densities, entropies, diagnostics
include("Hamiltonians.jl")  # QuadraticHamiltonian + propagator
include("Channels.jl")      # dissipation / measurement channels
include("Trajectory.jl")    # evolution, trajectory engine, ensemble runner
include("CorrelationLindblad.jl") # deterministic mixed-state Lindblad dynamics

# --- Majorana / BdG covariance layer (stage 2; unchanged) ---
include("Quadratic.jl")
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: PASS for the new `CorrelationLindblad` RHS assertions and no regressions.

- [ ] **Step 6: Commit**

```bash
git add src/CorrelationLindblad.jl src/GaussianFermions.jl test/runtests.jl
git commit -m "Add CorrelationLindblad RHS"
```

---

### Task 2: Dense Time Evolution and Steady State

**Files:**
- Modify: `src/CorrelationLindblad.jl`
- Test: `test/runtests.jl`

- [ ] **Step 1: Add failing evolution and steady-state tests**

Append these assertions to the `"CorrelationLindblad"` testset:

```julia
        H2 = hopping(4; pbc=true)
        s_unitary = CorrelationState(SlaterState(L=4, N=2, config="Z2"))
        s_det = copy(s_unitary)
        evolve!(s_unitary, H2, 0.3)
        evolve!(s_det, CorrelationLindblad(H2), 0.3)
        @test correlation_matrix(s_det) ≈ correlation_matrix(s_unitary) atol = 1e-10

        loss_ss = steadystate(loss_lind)
        @test density(loss_ss) ≈ [0.0] atol = 1e-10
        filled = CorrelationState([1.0])
        evolve!(filled, loss_lind, 20.0)
        @test density(filled)[1] < 1e-5

        gain_ss = steadystate(gain_lind)
        @test density(gain_ss) ≈ [1.0] atol = 1e-10
        empty = CorrelationState([0.0])
        evolve!(empty, gain_lind, 20.0)
        @test density(empty)[1] > 1 - 1e-5

        balanced_ss = steadystate(both_lind)
        @test density(balanced_ss) ≈ [0.25] atol = 1e-10

        cdecay = CorrelationState(ComplexF64[0.5 0.25; 0.25 0.5])
        evolve!(cdecay, deph_lind, 5.0)
        @test density(cdecay) ≈ [0.5, 0.5] atol = 1e-10
        @test abs(correlation(cdecay, 1, 2)) < 0.25
        @test_throws ArgumentError steadystate(deph_lind)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL with `UndefVarError: steadystate not defined` or `MethodError: no method matching evolve!(::CorrelationState, ::CorrelationLindblad, ...)`.

- [ ] **Step 3: Implement cached affine generator**

Add this code to `src/CorrelationLindblad.jl` after `lindblad_rhs`:

```julia
function _affine_generator(Liouv::CorrelationLindblad)
    if Liouv.cache !== nothing
        return Liouv.cache
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
    Liouv.cache = (M, b)
    Liouv.cache
end

function _hermitian_correlation(C::AbstractMatrix)
    Cs = (Matrix{ComplexF64}(C) + Matrix{ComplexF64}(C)') / 2
    Hermitian(Cs)
end
```

- [ ] **Step 4: Implement `evolve!`, `evolve`, and `steadystate`**

Add this code to `src/CorrelationLindblad.jl` after `_hermitian_correlation`:

```julia
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
    if check_unique && rank(M; atol) < size(M, 1)
        throw(ArgumentError("CorrelationLindblad steady state is not unique; the dense generator is singular"))
    end

    x = try
        -(M \ b)
    catch err
        throw(ArgumentError("CorrelationLindblad steady-state solve failed: $(err)"))
    end

    L = size(Liouv.h, 1)
    C = Matrix(_hermitian_correlation(reshape(x, L, L)))
    residual = norm(lindblad_rhs(Liouv, C))
    scale = max(norm(C), one(Float64))
    residual ≤ atol + rtol * scale ||
        throw(ArgumentError("CorrelationLindblad steady-state residual $residual exceeds tolerance"))

    vals = real.(eigvals(Hermitian(C)))
    all(vals .≥ -sqrt(atol)) && all(vals .≤ 1 + sqrt(atol)) ||
        throw(ArgumentError("CorrelationLindblad steady state is outside the physical occupation range"))

    CorrelationState(Hermitian(C))
end
```

- [ ] **Step 5: Run test to verify it passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: PASS for deterministic evolution and steady-state tests.

- [ ] **Step 6: Commit**

```bash
git add src/CorrelationLindblad.jl test/runtests.jl
git commit -m "Add dense CorrelationLindblad evolution"
```

---

### Task 3: Channel Adapters

**Files:**
- Modify: `src/CorrelationLindblad.jl`
- Test: `test/runtests.jl`

- [ ] **Step 1: Add failing channel adapter tests**

Append these assertions to the `"CorrelationLindblad"` testset:

```julia
        channel_loss = CorrelationLindblad(zeros(ComplexF64, 1, 1), [loss(1, 1; γ=0.7)])
        @test lindblad_rhs(channel_loss, ComplexF64[1;;]) ≈ lindblad_rhs(loss_lind, ComplexF64[1;;])

        channel_gain = CorrelationLindblad(zeros(ComplexF64, 1, 1), [gain(1, 1; γ=0.4)])
        @test lindblad_rhs(channel_gain, ComplexF64[0;;]) ≈ lindblad_rhs(gain_lind, ComplexF64[0;;])

        occ_ch = dephasing(1, 2; γ=0.5)
        hole_ch = HoleMonitor(QuasiMode([1], ComplexF64[1], 2); γ=0.5)
        occ_lind = CorrelationLindblad(zeros(ComplexF64, 2, 2), [occ_ch])
        hole_lind = CorrelationLindblad(zeros(ComplexF64, 2, 2), [hole_ch])
        @test lindblad_rhs(occ_lind, Ccoh) ≈ lindblad_rhs(deph_lind, Ccoh)
        @test lindblad_rhs(hole_lind, Ccoh) ≈ lindblad_rhs(deph_lind, Ccoh)

        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 2, 2), [loss(1, 3; γ=1.0)])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: FAIL with `MethodError: no method matching CorrelationLindblad(::Matrix, ::Vector...)`.

- [ ] **Step 3: Implement channel adapter constructors**

Add this code to `src/CorrelationLindblad.jl` after the keyword constructors:

```julia
function CorrelationLindblad(h::AbstractMatrix, channels)
    L = size(h, 1)
    loss_ops = Vector{Vector{ComplexF64}}()
    gain_ops = Vector{Vector{ComplexF64}}()
    dephasing_ops = Vector{Tuple{Matrix{ComplexF64},Float64}}()

    for ch in channels
        ch.mode.L == L || throw(ArgumentError("channel mode length $(ch.mode.L) does not match system size $L"))
        v = Vector{ComplexF64}(vector(ch.mode))
        if ch isa Loss
            push!(loss_ops, sqrt(ch.γ) .* v)
        elseif ch isa Gain
            push!(gain_ops, sqrt(ch.γ) .* v)
        elseif ch isa Union{OccupationMonitor,HoleMonitor}
            push!(dephasing_ops, (conj(v * v'), ch.γ))
        else
            throw(ArgumentError("unsupported channel type for CorrelationLindblad: $(typeof(ch))"))
        end
    end

    CorrelationLindblad(h; loss_ops, gain_ops, dephasing_ops)
end

CorrelationLindblad(H::QuadraticHamiltonian, channels) =
    CorrelationLindblad(Matrix(H.h), channels)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
julia --project=. test/runtests.jl
```

Expected: PASS for channel adapter tests.

- [ ] **Step 5: Commit**

```bash
git add src/CorrelationLindblad.jl test/runtests.jl
git commit -m "Add CorrelationLindblad channel adapters"
```

---

### Task 4: Documentation and Full Verification

**Files:**
- Modify: `README.md`
- Test: full test suite and examples

- [ ] **Step 1: Add README example**

Replace `README.md` with:

````markdown
# GaussianFermions.jl

Simulation of fermionic Gaussian states under unitary, Lindblad, and quantum-trajectory
(monitored) dynamics. Supports number-conserving free-fermion states as well as general
quadratic/Majorana states.

## Installation

```julia
pkg> add https://github.com/jayren3996/GaussianFermions.jl
```

## Deterministic Correlation-Matrix Lindblad Dynamics

```julia
using GaussianFermions

L = 16
H = hopping(L; pbc=true)

loss1 = zeros(ComplexF64, L)
loss1[1] = sqrt(0.2)

lind = CorrelationLindblad(H; loss_ops=[loss1])
state = CorrelationState(SlaterState(L=L, N=8, config="Z2"))

evolve!(state, lind, 1.0)
density(state)
```

For finite systems with a unique attractive state, solve directly:

```julia
steady = steadystate(CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                        loss_ops=[ComplexF64[sqrt(0.6)]],
                                        gain_ops=[ComplexF64[sqrt(0.2)]]))
density(steady)  # [0.25]
```
````

- [ ] **Step 2: Run full package tests**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS.

- [ ] **Step 3: Run examples**

Run:

```bash
julia --project=. example/FreeFermion.jl
julia --project=. example/MI.jl
julia --project=. example/Dephase.jl
```

Expected: each command exits with code 0. `example/Dephase.jl` prints trajectory-vs-Lindblad comparison norms.

- [ ] **Step 4: Inspect diff for accidental `Quadratic.jl` changes**

Run:

```bash
git diff -- src/Quadratic.jl
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Document CorrelationLindblad usage"
```

---

## Final Verification

- [ ] Run:

```bash
git status --short
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. example/FreeFermion.jl
julia --project=. example/MI.jl
julia --project=. example/Dephase.jl
```

- [ ] Expected:

```text
git status --short
# no output

Pkg.test()
# tests passed

examples
# each exits 0
```

- [ ] Final response should report the commits made, the verification commands run, and any residual limitations: dense generator only, unique steady-state requirement, and no Stage 2 changes.
