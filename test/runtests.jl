using GaussianFermions
using Test, LinearAlgebra, Random

isorthonormal(s::SlaterState) = isapprox(s.B' * s.B, I(particle_number(s)); atol=1e-10)
spectrum_ok(s) = all(-1e-9 .≤ real.(eigvals(Hermitian(correlation_matrix(s)))) .≤ 1 + 1e-9)

@testset "GaussianFermions.jl" begin

    @testset "States & constructors" begin
        s = SlaterState([1, 3], 4)
        @test particle_number(s) == 2
        @test nmodes(s) == 4
        @test density(s) ≈ [1, 0, 1, 0]
        @test isorthonormal(s)
        @test correlation_matrix(s) ≈ diagm([1, 0, 1, 0])
        @test correlation(s, 1, 1) ≈ 1
        @test correlation(s, 2, 2) ≈ 0
        @test SlaterState([true, false, true, false]).B == s.B

        @test density(SlaterState(L=4, N=2, config="left"))   ≈ [1, 1, 0, 0]
        @test density(SlaterState(L=4, N=2, config="right"))  ≈ [0, 0, 1, 1]
        @test density(SlaterState(L=4, N=2, config="center")) ≈ [0, 1, 1, 0]
        @test density(SlaterState(L=4, N=2, config="Z2"))     ≈ [1, 0, 1, 0]
        Random.seed!(1)
        @test particle_number(SlaterState(L=8, N=3, config="random")) == 3
        @test particle_number(SlaterState(8, 3)) == 3            # random Slater
        @test isorthonormal(SlaterState(8, 3))

        # ground state of a hopping chain
        h = Hermitian(Matrix(hopping(6).h))
        g = SlaterState(h, 3)
        @test particle_number(g) == 3
        @test isorthonormal(g)
    end

    @testset "Mixed states & conversions" begin
        s = SlaterState([1, 3], 4)
        cs = CorrelationState(s)
        @test correlation_matrix(cs) ≈ correlation_matrix(s)
        @test ispure(cs)
        @test particle_number(cs) ≈ 2
        @test SlaterState(cs).B'SlaterState(cs).B ≈ I(2)        # pure -> orbital round trip

        th = thermalstate(Hermitian(Matrix(hopping(6).h)); β=1.0)
        @test !ispure(th)
        @test all(0 .≤ density(th) .≤ 1)
        @test spectrum_ok(th)
        @test density(maximally_mixed(4)) ≈ fill(0.5, 4)
        @test_throws ArgumentError SlaterState(th)              # not pure
    end

    @testset "Observables" begin
        s = SlaterState([1, 3], 4)
        @test entanglement_entropy(s, 1:2) ≈ 0 atol = 1e-10
        @test mutual_information(s, [1], [3]) ≈ 0 atol = 1e-10
        @test tripartite_information(s, [1], [2], [3]) ≈ 0 atol = 1e-10
        @test number_variance(s, 1:2) ≈ 0 atol = 1e-10
        @test purity(s) ≈ 1 atol = 1e-10
        @test parity(s) ≈ 1 atol = 1e-10               # N=2 even
        @test parity(SlaterState([1], 2)) ≈ -1 atol = 1e-10  # N=1 odd

        # one particle in a 50/50 superposition -> S = log 2
        s2 = SlaterState([1], 2)
        evolve!(s2, ComplexF64[1 1; 1 -1] / sqrt(2))
        @test entanglement_entropy(s2, [1]) ≈ log(2) atol = 1e-8
        @test renyi_entropy(s2, [1]; α=1) ≈ entanglement_entropy(s2, [1])
        @test renyi_entropy(s2, [1]; α=2) ≈ -log(0.5^2 + 0.5^2) atol = 1e-8
        @test renyi_entropy(s2, [1]; α=Inf) ≈ -log(0.5) atol = 1e-8
        @test length(entanglement_spectrum(s2, [1])) == 1
        # mixed-state observables go through the eigenvalue path too
        @test entanglement_entropy(CorrelationState(s2), [1]) ≈ log(2) atol = 1e-8
    end

    @testset "Hamiltonian & evolution" begin
        L = 6
        H = hopping(L; pbc=true)
        U = propagator(H, 0.3)
        @test U * U' ≈ I(L)
        @test propagator(H, 0.3) === propagator(H, 0.3)   # cached

        s = SlaterState(L=L, N=3, config="Z2")
        n0 = particle_number(s)
        evolve!(s, H, 0.3)
        @test particle_number(s) == n0
        @test isorthonormal(s)

        # pure and mixed evolution agree
        sp = SlaterState(L=L, N=3, config="Z2")
        sm = CorrelationState(sp)
        evolve!(sp, H, 0.7); evolve!(sm, H, 0.7)
        @test correlation_matrix(sp) ≈ correlation_matrix(sm)
        @test particle_number(sm) ≈ 3
        @test spectrum_ok(sm)

        # local gate matches full application
        a = SlaterState([1], 2); b = SlaterState([1], 2)
        g = ComplexF64[1 1; 1 -1] / sqrt(2)
        evolve!(a, g); apply!(g, b, [1, 2])
        @test a.B ≈ b.B
    end

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

    @testset "Channels & trajectories" begin
        Random.seed!(42)
        L = 8
        H = hopping(L; pbc=true)

        # occupation monitoring conserves N
        s = SlaterState(L=L, N=4, config="Z2")
        for _ in 1:50; step!(s, H, [dephasing(i, L; γ=1.0) for i in 1:L], 0.05); end
        @test particle_number(s) == 4
        @test isorthonormal(s)
        @test all(-1e-9 .≤ density(s) .≤ 1 + 1e-9)

        # hole monitoring conserves N
        sh = SlaterState(L=L, N=4, config="Z2")
        for _ in 1:30; step!(sh, H, [HoleMonitor(QuasiMode([i], ComplexF64[1], L); γ=1.0) for i in 1:L], 0.05); end
        @test particle_number(sh) == 4
        @test isorthonormal(sh)

        # loss -> vacuum (N non-increasing)
        sl = SlaterState(L=L, N=4, config="Z2")
        for _ in 1:100; step!(sl, H, [loss(i, L; γ=1.0) for i in 1:L], 0.05); end
        @test particle_number(sl) == 0

        # gain -> N non-decreasing
        sg = SlaterState(L=L, N=2, config="left")
        for _ in 1:60; step!(sg, H, [gain(i, L; γ=1.0) for i in 1:L], 0.05); end
        @test particle_number(sg) ≥ 2
        @test isorthonormal(sg)
    end

    @testset "Continuous monitoring" begin
        Random.seed!(7)
        L = 8
        H = hopping(L; pbc=true)
        s = SlaterState(L=L, N=4, config="Z2")
        chans = [dephasing(i, L; γ=0.5) for i in 1:L]
        for _ in 1:50
            evolve!(s, H, 0.05)
            step_diffusive!(s, chans, 0.05)
        end
        @test particle_number(s) == 4
        @test isorthonormal(s)
    end

    @testset "No-click / non-Hermitian" begin
        Random.seed!(3)
        L = 8
        H = hopping(L; pbc=true)
        s = SlaterState(L=L, N=4, config="Z2")
        G = effective_hamiltonian(H, [loss(i, L; γ=1.0) for i in 1:L])
        w = 1.0
        for _ in 1:20
            wk = evolve_noclick!(s, G, 0.05)
            @test 0 ≤ wk ≤ 1 + 1e-9
            w *= wk
        end
        @test w ≤ 1
        @test isorthonormal(s)
    end

    @testset "Measurement & feedback" begin
        Random.seed!(5)
        L = 4
        @test measure!(QuasiMode([1], ComplexF64[1], L), SlaterState([1, 3], L)) == true
        @test measure!(QuasiMode([2], ComplexF64[1], L), SlaterState([1, 3], L)) == false

        s = SlaterState(L=L, N=2, config="Z2")
        fb = Feedback(dephasing(1, L; γ=0.5), ComplexF64[1;;], ComplexF64[1;;])
        @test apply!(fb, s, 0.1) isa Bool
        @test isorthonormal(s)
    end

    @testset "Ensemble runner" begin
        Random.seed!(11)
        L = 8
        H = hopping(L; pbc=true)
        res = ensemble(() -> SlaterState(L=L, N=4, config="Z2"), H,
                       [dephasing(i, L; γ=0.5) for i in 1:L];
                       ntraj=24, tspan=1.0, dt=0.1,
                       observables=(n=density, S=s -> entanglement_entropy(s, 1:L÷2)))
        @test length(res.saveat) == 10
        @test res.ntraj == 24
        @test length(res.mean.n[end]) == L              # density is a vector observable
        @test res.mean.S[end] ≥ 0                       # entropy non-negative
        @test all(0 .≤ res.mean.n[end] .≤ 1)
        # trajectory-averaged total density ≈ conserved N (occupation monitoring conserves N)
        @test sum(res.mean.n[end]) ≈ 4 atol = 1e-6
    end

end
