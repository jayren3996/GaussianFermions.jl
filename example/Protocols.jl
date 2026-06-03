using GaussianFermions, LinearAlgebra, Random

#--------------------------------------------------------------------------------
# Three unravelings of the same occupation-monitoring Lindbladian D[√γ nᵢ] on a
# hopping chain: projective measurement, quantum-jump (MCWF), and quantum-state
# diffusion (QSD / weak measurement).
#
#   • LINEAR observables (site density) agree across all three and with the
#     deterministic CorrelationLindblad — convention matching: channel/QSD rate γ,
#     projective rate γ/2 (projective nᵢ at rate γ_p unravels D[√(2γ_p) nᵢ]).
#   • ENTANGLEMENT is trajectory-nonlinear and unraveling-dependent: the protocols
#     do NOT give identical entropies, though each shows monitoring-induced
#     suppression. "Robustness" is phenomenological, not value-for-value.
#--------------------------------------------------------------------------------

function mcwf_step!(s, U, channels, dt; rng=Random.default_rng())
    evolve!(s, U)
    for ch in channels
        p, work = jump_rate(ch, s, dt)
        if rand(rng) < p
            apply_click!(ch, s, work); normalize!(s)
        else
            apply_noclick!(ch, s, dt)
        end
    end
    s
end

function lindblad_density(; L=12, γ=1.0, dt=0.1, T=4.0)
    lind  = CorrelationLindblad(hopping(L; pbc=false), [dephasing(i, L; γ=γ) for i in 1:L])
    state = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))
    for _ in 1:round(Int, T / dt); evolve!(state, lind, dt); end
    density(state)
end

function mcwf_density(; L=12, γ=1.0, dt=0.1, T=4.0, ntraj=400, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=false), dt); chans = [dephasing(i, L; γ=γ) for i in 1:L]
    acc = zeros(L)
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="left")
        for _ in 1:round(Int, T / dt); mcwf_step!(s, U, chans, dt; rng); end
        acc .+= density(s)
    end
    acc ./ ntraj
end

function qsd_density(; L=12, γ=1.0, dt=0.1, T=4.0, ntraj=400, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=false), dt); modes = [QuasiMode([i], ComplexF64[1], L) for i in 1:L]
    acc = zeros(L)
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="left")
        for _ in 1:round(Int, T / dt)
            evolve!(s, U)
            for qm in modes
                α = randn(rng) * sqrt(γ * dt) + (2 * density(s, qm) - 1) * γ * dt
                weak_measure!(s, qm, α)
            end
        end
        acc .+= density(s)
    end
    acc ./ ntraj
end

function projective_density(; L=12, γ=1.0, dt=0.1, T=4.0, ntraj=400, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=false), dt); modes = [QuasiMode([i], ComplexF64[1], L) for i in 1:L]
    acc = zeros(L)
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="left")
        for _ in 1:round(Int, T / dt)
            evolve!(s, U)
            for qm in modes
                rand(rng) < 0.5 * γ * dt && measure!(qm, s; rng)   # γ/2 → D[√γ n]
            end
        end
        acc .+= density(s)
    end
    acc ./ ntraj
end

# --- nonlinear observable: half-chain entanglement per unraveling (L=16 ring) ---
function mcwf_entropy(; L=16, γ=1.0, dt=0.1, tspan=12.0, ntraj=24, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=true), dt); chans = [dephasing(i, L; γ=γ) for i in 1:L]
    nsteps = round(Int, tspan / dt); tail = nsteps ÷ 4; Σ = 0.0; n = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
            k > nsteps - tail && (Σ += entanglement_entropy(s, 1:(L ÷ 2)); n += 1)
        end
    end
    Σ / n
end
function qsd_entropy(; L=16, γ=1.0, dt=0.1, tspan=12.0, ntraj=24, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=true), dt); modes = [QuasiMode([i], ComplexF64[1], L) for i in 1:L]
    nsteps = round(Int, tspan / dt); tail = nsteps ÷ 4; Σ = 0.0; n = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            evolve!(s, U)
            for qm in modes
                α = randn(rng) * sqrt(γ * dt) + (2 * density(s, qm) - 1) * γ * dt
                weak_measure!(s, qm, α)
            end
            k > nsteps - tail && (Σ += entanglement_entropy(s, 1:(L ÷ 2)); n += 1)
        end
    end
    Σ / n
end
function projective_entropy(; L=16, γ=1.0, dt=0.1, tspan=12.0, ntraj=24, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=true), dt); modes = [QuasiMode([i], ComplexF64[1], L) for i in 1:L]
    nsteps = round(Int, tspan / dt); tail = nsteps ÷ 4; Σ = 0.0; n = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            evolve!(s, U)
            for qm in modes
                rand(rng) < 0.5 * γ * dt && measure!(qm, s; rng)
            end
            k > nsteps - tail && (Σ += entanglement_entropy(s, 1:(L ÷ 2)); n += 1)
        end
    end
    Σ / n
end

function main()
    ref = lindblad_density()
    println("-- linear observable (density) --")
    println("Lindblad   density[1:4] = ", round.(ref[1:4]; digits=3))
    println("MCWF       ‖Δ‖ = ", round(norm(mcwf_density() - ref); digits=3))
    println("QSD        ‖Δ‖ = ", round(norm(qsd_density() - ref); digits=3))
    println("Projective ‖Δ‖ = ", round(norm(projective_density() - ref); digits=3))
    println("-- nonlinear observable (half-chain entanglement, L=16 ring) --")
    for γ in (0.5, 1.0, 2.0)
        println("γ=$γ  MCWF=", round(mcwf_entropy(; γ=γ); digits=3),
                "  QSD=", round(qsd_entropy(; γ=γ); digits=3),
                "  projective=", round(projective_entropy(; γ=γ); digits=3))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
