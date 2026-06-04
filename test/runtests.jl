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

    @testset "Majorana/BdG foundation" begin
        occ = [1, 0, 1, 0]
        ms = MajoranaState(occ)
        Γ = covariance_matrix(ms)
        @test nmodes(ms) == 4
        @test size(Γ) == (8, 8)
        @test eltype(Γ) == Float64
        @test Γ ≈ -transpose(Γ)
        @test normal_correlation(ms) ≈ diagm(ComplexF64.(occ))
        @test anomalous_correlation(ms) ≈ zeros(ComplexF64, 4, 4)

        slater = SlaterState(L=4, N=2, config="Z2")
        from_slater = MajoranaState(slater)
        @test normal_correlation(from_slater) ≈ correlation_matrix(slater)
        @test anomalous_correlation(from_slater) ≈ zeros(ComplexF64, 4, 4) atol = 1e-12

        # O(1) density(s, i) agrees with the full profile; ispure detects purity;
        # and a floating-point element type is preserved by the constructor.
        @test [density(from_slater, i) for i in 1:4] ≈ density(from_slater)
        @test ispure(from_slater)
        mixed = MajoranaState(thermalstate(Hermitian(Matrix(hopping(4).h)); β=0.5))
        @test !ispure(mixed)
        Γ32 = Float32.(covariance_matrix(from_slater))
        @test eltype(covariance_matrix(MajoranaState(Γ32; check=false))) == Float32

        cmat = ComplexF64[
            0.6   0.2im
           -0.2im 0.4
        ]
        corr = CorrelationState(cmat)
        from_corr = MajoranaState(corr)
        @test normal_correlation(from_corr) ≈ correlation_matrix(corr)

        @test covariance_matrix(ms) !== ms.Gamma
        Γcopy = covariance_matrix(ms)
        Γcopy[1, 2] = 99
        @test covariance_matrix(ms)[1, 2] != 99

        @test_throws ArgumentError MajoranaState(zeros(3, 3))
        @test_throws ArgumentError MajoranaState(zeros(2, 3))
        bad = zeros(4, 4); bad[1, 2] = 0.1
        @test_throws ArgumentError MajoranaState(bad)
        nonfinite = zeros(4, 4); nonfinite[1, 2] = Inf; nonfinite[2, 1] = -Inf
        @test_throws ArgumentError MajoranaState(nonfinite)
        unphysical = zeros(4, 4); unphysical[1, 3] = 2.0; unphysical[3, 1] = -2.0
        @test_throws ArgumentError MajoranaState(unphysical)

        K = zeros(Float64, 4, 4)
        K[1, 3] = 0.7; K[3, 1] = -0.7
        bdg = BdGHamiltonian(K)
        O = propagator(bdg, 0.2)
        @test O * transpose(O) ≈ I(4) atol = 1e-12
        @test propagator(bdg, 0.2) === propagator(bdg, 0.2)
        @test_throws ArgumentError BdGHamiltonian(zeros(3, 3))
        @test_throws ArgumentError BdGHamiltonian(zeros(2, 3))
        notanti = zeros(4, 4); notanti[1, 2] = 0.1
        @test_throws ArgumentError BdGHamiltonian(notanti)
        badK = zeros(4, 4); badK[1, 2] = NaN; badK[2, 1] = -NaN
        @test_throws ArgumentError BdGHamiltonian(badK)

        Hnc = hopping(4; pbc=true)
        cstate = CorrelationState(SlaterState(L=4, N=2, config="Z2"))
        mstate = MajoranaState(cstate)
        evolve!(cstate, Hnc, 0.37)
        evolve!(mstate, BdGHamiltonian(Hnc), 0.37)
        @test normal_correlation(mstate) ≈ correlation_matrix(cstate) atol = 1e-10

        evolved_copy = evolve(mstate, BdGHamiltonian(Hnc), 0.1)
        @test evolved_copy !== mstate
        @test covariance_matrix(evolved_copy) != covariance_matrix(mstate)

        @test_throws ArgumentError evolve!(MajoranaState([1, 0]), Matrix(I, 6, 6))

        prod_state = MajoranaState([1, 0, 1, 0])
        @test density(prod_state) ≈ [1, 0, 1, 0]
        @test particle_number(prod_state) ≈ 2
        @test particle_number(prod_state, [1, 2]) ≈ 1
        @test entanglement_entropy(prod_state, [1, 2]) ≈ 0 atol = 1e-12
        @test renyi_entropy(prod_state, [1, 2]; α=2) ≈ 0 atol = 1e-12
        @test mutual_information(prod_state, [1], [2]) ≈ 0 atol = 1e-12
        @test tripartite_information(prod_state, [1], [2], [3]) ≈ 0 atol = 1e-12
        @test entanglement_spectrum(prod_state, [1, 2]) ≈ [1, 1] atol = 1e-12

        ent_slater = SlaterState([1], 2)
        evolve!(ent_slater, ComplexF64[1 1; 1 -1] / sqrt(2))
        ent_majorana = MajoranaState(ent_slater)
        @test density(ent_majorana) ≈ density(ent_slater)
        @test entanglement_entropy(ent_majorana, [1]) ≈ entanglement_entropy(ent_slater, [1]) atol = 1e-10
        @test renyi_entropy(ent_majorana, [1]; α=Inf) ≈ renyi_entropy(ent_slater, [1]; α=Inf) atol = 1e-10

        A = zeros(ComplexF64, 2, 2)
        B = zeros(ComplexF64, 2, 2)
        B[1, 2] = 0.5
        B[2, 1] = -0.5
        paired = MajoranaState([0, 0])
        evolve!(paired, BdGHamiltonian(A, B), 0.4)
        @test norm(anomalous_correlation(paired)) > 1e-3
        @test_throws ArgumentError BdGHamiltonian(A, ComplexF64[0 1; 1 0])

        @test !isdefined(GaussianFermions, :covariancematrix)
        @test !isdefined(GaussianFermions, :fermioncorrelation)
        @test !isdefined(GaussianFermions, :majoranaform)
        @test !isdefined(GaussianFermions, :quadraticlindblad)
        @test !isdefined(GaussianFermions, :lindblad_evo)
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

        # profile aliases and the entanglement-Hamiltonian spectrum
        @test density_profile(s) == density(s)
        @test correlation_profile(s, 1) == [correlation(s, 1, j) for j in 1:nmodes(s)]
        @test entanglement_hamiltonian_spectrum(s2, [1]) ≈ [0.0] atol = 1e-8   # ν=1/2 ⇒ ε=0
    end

    @testset "Degenerate-size edge cases" begin
        # Fully filled (N = L): product state, zero entanglement, odd parity for N=3.
        full = SlaterState(collect(1:3), 3)
        @test particle_number(full) == 3
        @test density(full) ≈ [1, 1, 1]
        @test entanglement_entropy(full, [1, 2]) ≈ 0 atol = 1e-12
        @test parity(full) ≈ -1 atol = 1e-10

        # Empty system (L = 0): observables are empty / trivial and do not error.
        e = SlaterState(Int[], 0)
        @test nmodes(e) == 0
        @test particle_number(e) == 0
        @test isempty(density(e))
        @test entanglement_entropy(e, Int[]) ≈ 0 atol = 1e-12

        me = MajoranaState(Int[])
        @test nmodes(me) == 0
        @test isempty(density(me))
        @test size(covariance_matrix(me)) == (0, 0)
        @test ispure(me)                               # vacuously pure
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

        # Gate wrapper and the vector-of-Gates apply! match raw matrix applications
        g1 = ComplexF64[1 1; 1 -1] / sqrt(2); g2 = ComplexF64[0 1; 1 0]
        gA = SlaterState([1], 4); gB = SlaterState([1], 4)
        apply!([Gate(g1, [1, 2]), Gate(g2, [3, 4])], gA)
        apply!(g1, gB, [1, 2]); apply!(g2, gB, [3, 4])
        @test gA.B ≈ gB.B
        @test apply!(Gate(g1, [1, 2]), SlaterState([1], 4)).B ≈ apply!(g1, SlaterState([1], 4), [1, 2]).B
    end

    @testset "Model constructors" begin
        Hssh = ssh_chain(4; t1=1.0, t2=0.25, pbc=false)
        @test Matrix(Hssh) ≈ ComplexF64[
            0    1.0  0     0
            1.0  0    0.25  0
            0    0.25 0     1.0
            0    0    1.0   0
        ]

        Hssh_pbc = ssh_chain(4; t1=1.0, t2=0.25, pbc=true)
        @test Matrix(Hssh_pbc)[4, 1] ≈ 0.25
        @test Matrix(Hssh_pbc)[1, 4] ≈ 0.25
        @test_throws ArgumentError ssh_chain(3; pbc=true)

        Haa = aubry_andre_chain(3; J=2.0, λ=0.5, β=0.25, ϕ=0.1, pbc=false)
        expected_aa = Matrix(hopping(3; J=2.0, pbc=false).h)
        expected_aa .+= diagm(ComplexF64[0.5 * cos(2π * 0.25 * j + 0.1) for j in 1:3])
        @test Matrix(Haa) ≈ expected_aa
        @test_throws ArgumentError aubry_andre_chain(1; pbc=true)

        Hk = kitaev_chain(4; t=1.0, Δ=0.4, μ=0.2, pbc=false)
        A = zeros(ComplexF64, 4, 4)
        B = zeros(ComplexF64, 4, 4)
        for j in 1:4
            A[j, j] = -0.2
        end
        for j in 1:3
            A[j, j+1] = -1.0
            A[j+1, j] = -1.0
            B[j, j+1] = 0.4
            B[j+1, j] = -0.4
        end
        @test Matrix(Hk) ≈ Matrix(BdGHamiltonian(A, B))

        Hk_pbc = kitaev_chain(4; t=1.0, Δ=0.4, μ=0.2, pbc=true)
        A[4, 1] = -1.0
        A[1, 4] = -1.0
        B[4, 1] = 0.4
        B[1, 4] = -0.4
        @test Matrix(Hk_pbc) ≈ Matrix(BdGHamiltonian(A, B))
        @test_throws ArgumentError kitaev_chain(1; pbc=true)
        @test_throws ArgumentError kitaev_chain(2; pbc=true)

        @test_throws ArgumentError ssh_chain(0)
        @test_throws ArgumentError aubry_andre_chain(0)
        @test_throws ArgumentError kitaev_chain(0)
    end

    @testset "Spectral helpers" begin
        Hssh = ssh_chain(4; t1=1.0, t2=0.25)
        @test energy_spectrum(Hssh) ≈ sort(real.(eigvals(Hermitian(Matrix(Hssh)))))

        Hk = kitaev_chain(4; t=1.0, Δ=0.4, μ=0.2)
        @test energy_spectrum(Hk) ≈ quasiparticle_energies(Hk)

        Hbloch(k) = ComplexF64[
            0                  1 + exp(-im * k)
            1 + exp(im * k)    0
        ]
        bands = bloch_bands(Hbloch, [0.0, π])
        @test size(bands) == (2, 2)
        @test bands[1, :] ≈ [-2.0, 2.0]
        @test bands[2, :] ≈ [0.0, 0.0] atol = 1e-12

        Hbdg_zero(k) = BdGHamiltonian([0.0 k; -k 0.0])
        @test energy_spectrum(Hbdg_zero(0.0)) ≈ [0.0]
        @test quasiparticle_energies(Hbdg_zero(0.0)) ≈ [0.0]
        @test bloch_bands(Hbdg_zero, [-1.0, 0.0, 1.0]) ≈ reshape([1.0, 0.0, 1.0], 3, 1)

        bad_Hk(k) = k == 0 ? zeros(ComplexF64, 2, 2) : zeros(ComplexF64, 3, 3)
        @test_throws ArgumentError bloch_bands(bad_Hk, [0.0, 1.0])
    end

    @testset "CorrelationLindblad" begin
        unitmode(L, i) = (v = zeros(ComplexF64, L); v[i] = 1; v)
        @test isdefined(GaussianFermions, :steadystate)

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
        @test_throws ArgumentError lindblad_rhs(lind_h, zeros(ComplexF64, 2, 2))

        loss_lind = CorrelationLindblad(zeros(ComplexF64, 1, 1); loss_ops=[sqrt(0.7) * unitmode(1, 1)])
        @test lindblad_rhs(loss_lind, ComplexF64[1;;]) ≈ ComplexF64[-0.7;;]
        @test lindblad_rhs(loss_lind, CorrelationState([1.0])) ≈ ComplexF64[-0.7;;]

        gain_lind = CorrelationLindblad(zeros(ComplexF64, 1, 1); gain_ops=[sqrt(0.4) * unitmode(1, 1)])
        @test lindblad_rhs(gain_lind, ComplexF64[0;;]) ≈ ComplexF64[0.4;;]
        @test lindblad_rhs(gain_lind, ComplexF64[1;;]) ≈ ComplexF64[0;;] atol = 1e-12

        mutated_lind = CorrelationLindblad(zeros(ComplexF64, 1, 1); loss_ops=[sqrt(0.7) * unitmode(1, 1)])
        evolve!(CorrelationState([1.0]), mutated_lind, 0.1)
        mutated_lind.source[1, 1] = 0.7
        mutated_lind.damping[1, 1] = 0.7
        mutated_state = CorrelationState([0.0])
        evolve!(mutated_state, mutated_lind, 1.0)
        @test density(mutated_state) ≈ [1 - exp(-0.7)] atol = 1e-10

        both_lind = CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                        loss_ops=[sqrt(0.6) * unitmode(1, 1)],
                                        gain_ops=[sqrt(0.2) * unitmode(1, 1)])
        @test lindblad_rhs(both_lind, ComplexF64[0.25;;]) ≈ ComplexF64[0;;] atol = 1e-12

        deph_lind = CorrelationLindblad(zeros(ComplexF64, 2, 2);
                                        dephasing_ops=[(unitmode(2, 1), 0.5)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 2, 2); dephasing_ops=[1])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 2, 2);
                                                       dephasing_ops=[(unitmode(2, 1), Inf)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1); loss_ops=[ComplexF64[NaN]])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1); gain_ops=[ComplexF64[Inf]])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1); dephasing_ops=[(ComplexF64[NaN], 1.0)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1); dephasing_ops=[(ComplexF64[Inf;;], 1.0)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1); loss_ops=[QuasiMode([1], ComplexF64[NaN], 1)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1); dephasing_ops=[(QuasiMode([1], ComplexF64[Inf], 1), 1.0)])
        Ccoh = ComplexF64[0.5 0.25; 0.25 0.5]
        dC = lindblad_rhs(deph_lind, Ccoh)
        @test diag(dC) ≈ [0, 0] atol = 1e-12
        @test dC[1, 2] ≈ -0.25 * 0.5 / 2 atol = 1e-12
        @test dC[2, 1] ≈ -0.25 * 0.5 / 2 atol = 1e-12

        channel_loss = CorrelationLindblad(zeros(ComplexF64, 1, 1), [loss(1, 1; γ=0.7)])
        @test lindblad_rhs(channel_loss, ComplexF64[1;;]) ≈ lindblad_rhs(loss_lind, ComplexF64[1;;])
        channel_loss_h = CorrelationLindblad(QuadraticHamiltonian(zeros(ComplexF64, 1, 1)),
                                             [loss(1, 1; γ=0.7)])
        @test lindblad_rhs(channel_loss_h, ComplexF64[1;;]) ≈ lindblad_rhs(channel_loss, ComplexF64[1;;])

        channel_gain = CorrelationLindblad(zeros(ComplexF64, 1, 1), [gain(1, 1; γ=0.4)])
        @test lindblad_rhs(channel_gain, ComplexF64[0;;]) ≈ lindblad_rhs(gain_lind, ComplexF64[0;;])
        @test lindblad_rhs(channel_gain, ComplexF64[1;;]) ≈ lindblad_rhs(gain_lind, ComplexF64[1;;])

        complex_loss = Loss(QuasiMode([1, 2], ComplexF64[1, im] / sqrt(2), 2); γ=0.3)
        channel_complex_loss = CorrelationLindblad(zeros(ComplexF64, 2, 2), [complex_loss])
        complex_loss_vec = sqrt(0.3) .* (ComplexF64[1, im] / sqrt(2))
        lowlevel_complex_loss = CorrelationLindblad(zeros(ComplexF64, 2, 2);
                                                   loss_ops=[complex_loss_vec])
        @test lindblad_rhs(channel_complex_loss, Ccoh) ≈ lindblad_rhs(lowlevel_complex_loss, Ccoh)

        occ_ch = dephasing(1, 2; γ=0.5)
        hole_ch = HoleMonitor(QuasiMode([1], ComplexF64[1], 2); γ=0.5)
        occ_lind = CorrelationLindblad(zeros(ComplexF64, 2, 2), [occ_ch])
        hole_lind = CorrelationLindblad(zeros(ComplexF64, 2, 2), [hole_ch])
        @test lindblad_rhs(occ_lind, Ccoh) ≈ lindblad_rhs(deph_lind, Ccoh)
        @test lindblad_rhs(hole_lind, Ccoh) ≈ lindblad_rhs(deph_lind, Ccoh)
        complex_mode = QuasiMode([1, 2], ComplexF64[1, im] / sqrt(2), 2)
        complex_monitor = OccupationMonitor(complex_mode; γ=0.3)
        monitor_lind = CorrelationLindblad(zeros(ComplexF64, 2, 2), [complex_monitor])
        low_level_monitor_lind = CorrelationLindblad(zeros(ComplexF64, 2, 2);
                                                     dephasing_ops=[(complex_mode, 0.3)])
        @test lindblad_rhs(monitor_lind, Ccoh) ≈ lindblad_rhs(low_level_monitor_lind, Ccoh)

        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 2, 2), [loss(1, 3; γ=1.0)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1), [loss(1, 1; γ=-1.0)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1), [gain(1, 1; γ=NaN)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 2, 2), [dephasing(1, 2; γ=Inf)])
        @test_throws ArgumentError CorrelationLindblad(zeros(ComplexF64, 1, 1), [nothing])

        v = ComplexF64[1, im] / sqrt(2)
        complex_deph = CorrelationLindblad(zeros(ComplexF64, 2, 2);
                                           dephasing_ops=[(v, 0.3)])
        Q = v * v'                              # standard projector (matches ⟨c⁺c⟩ convention)
        @test complex_deph.dephasing[1][2] ≈ Q

        H2 = hopping(4; pbc=true)
        s_unitary = CorrelationState(SlaterState(L=4, N=2, config="Z2"))
        s_det = copy(s_unitary)
        evolve!(s_unitary, H2, 0.3)
        evolve!(s_det, CorrelationLindblad(H2), 0.3)
        @test correlation_matrix(s_det) ≈ correlation_matrix(s_unitary) atol = 1e-10
        @test_throws ArgumentError evolve!(CorrelationState([1.0]), CorrelationLindblad(zeros(ComplexF64, 2, 2)), 0.1)

        loss_ss = steadystate(loss_lind)
        @test density(loss_ss) ≈ [0.0] atol = 1e-10
        filled = CorrelationState([1.0])
        evolve!(filled, loss_lind, 20.0)
        @test density(filled)[1] < 1e-5

        gain_ss = steadystate(gain_lind)
        @test density(gain_ss) ≈ [1.0] atol = 1e-10
        empty = CorrelationState([0.0])
        evolve!(empty, gain_lind, 30.0)
        @test density(empty)[1] > 1 - 1e-5

        balanced_ss = steadystate(both_lind)
        @test density(balanced_ss) ≈ [0.25] atol = 1e-10

        small_balanced = CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                             loss_ops=[sqrt(1e-12) * unitmode(1, 1)],
                                             gain_ops=[sqrt(1e-12) * unitmode(1, 1)])
        @test density(steadystate(small_balanced)) ≈ [0.5] atol = 1e-10

        large_balanced = CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                             loss_ops=[sqrt(1e8) * unitmode(1, 1)],
                                             gain_ops=[sqrt(1e8) * unitmode(1, 1)])
        @test density(steadystate(large_balanced)) ≈ [0.5] atol = 1e-10

        multiscale_balanced = CorrelationLindblad(zeros(ComplexF64, 2, 2);
                                                  loss_ops=[unitmode(2, 1), sqrt(1e-10) * unitmode(2, 2)],
                                                  gain_ops=[unitmode(2, 1), sqrt(1e-10) * unitmode(2, 2)])
        @test density(steadystate(multiscale_balanced)) ≈ [0.5, 0.5] atol = 1e-10

        # multi-mode steady state of a coupled (H ≠ 0) dissipative chain must equal the
        # long-time limit of evolve! (not just the analytic single-mode cases above).
        Hms = hopping(3; pbc=true)
        ms_lind = CorrelationLindblad(Hms; loss_ops=[sqrt(0.5) * unitmode(3, 1)],
                                           gain_ops=[sqrt(0.3) * unitmode(3, 3)])
        ms_ss = steadystate(ms_lind)
        ms_evolved = CorrelationState(SlaterState(L=3, N=2, config="Z2"))
        evolve!(ms_evolved, ms_lind, 120.0)
        @test correlation_matrix(ms_ss) ≈ correlation_matrix(ms_evolved) atol = 1e-6

        cdecay = CorrelationState(ComplexF64[0.5 0.25; 0.25 0.5])
        evolve!(cdecay, deph_lind, 5.0)
        @test density(cdecay) ≈ [0.5, 0.5] atol = 1e-10
        @test correlation(cdecay, 1, 2) ≈ 0.25 * exp(-0.5 * 5.0 / 2) atol = 1e-10
        @test_throws ArgumentError steadystate(deph_lind)
    end

    @testset "MajoranaLindblad" begin
        unitmode(L, i) = (v = zeros(ComplexF64, L); v[i] = 1; v)
        L = 3
        H = hopping(L; pbc=true)
        γ = 0.6
        vmode = normalize(ComplexF64[1, 1, 0])

        # Number-conserving sector must match CorrelationLindblad exactly: build the
        # same Dirac generator both ways, evolve a CorrelationState and the converted
        # MajoranaState, and compare normal correlations.
        loss = sqrt(γ) .* unitmode(L, 1)
        gain = sqrt(γ) .* unitmode(L, 2)
        corr_lind = CorrelationLindblad(H; loss_ops=[loss], gain_ops=[gain],
                                        dephasing_ops=[(vmode, γ)])
        maj_lind = MajoranaLindblad(H; loss_ops=[loss], gain_ops=[gain],
                                    dephasing_ops=[(vmode, γ)])

        c0 = CorrelationState(SlaterState(L=L, N=2, config="left"))
        m0 = MajoranaState(c0)
        for dt in (0.13, 0.13, 0.4)
            evolve!(c0, corr_lind, dt)
            evolve!(m0, maj_lind, dt)
        end
        @test normal_correlation(m0) ≈ correlation_matrix(c0) atol = 1e-9
        @test norm(anomalous_correlation(m0)) ≈ 0 atol = 1e-9   # number-conserving stays paired-free

        # rhs agrees with CorrelationLindblad (mapped through the C↔Γ bijection)
        mrhs = majorana_lindblad_rhs(maj_lind, MajoranaState(c0))
        @test mrhs ≈ -transpose(mrhs) atol = 1e-12              # real antisymmetric

        # steady state of a single loss+gain mode matches the analytic balance
        ss_maj = steadystate(MajoranaLindblad(chemical_potential(zeros(1));
                                              loss_ops=[ComplexF64[sqrt(0.6)]],
                                              gain_ops=[ComplexF64[sqrt(0.2)]]))
        @test density(ss_maj) ≈ [0.25] atol = 1e-8

        # Pairing: a jump operator with a creation component generates anomalous
        # correlations and keeps the state physical.
        lpair = zeros(ComplexF64, 2L)
        lpair[1] = 0.5; lpair[1+L] = -0.5im            # c₁
        lpair[2] = 0.5; lpair[2+L] = 0.5im             # c₂⁺
        lpair .*= sqrt(γ)
        pair_lind = MajoranaLindblad(BdGHamiltonian(H); majorana_ops=[lpair])
        ms = MajoranaState([0, 0, 0])
        evolve!(ms, pair_lind, 2.5)
        @test norm(anomalous_correlation(ms)) > 1e-3
        @test maximum(abs.(real.(eigvals(Hermitian(im .* covariance_matrix(ms)))))) ≤ 1 + 1e-8

        # input validation
        @test_throws ArgumentError majorana_lindblad_rhs(maj_lind, zeros(4, 4))
        @test_throws ArgumentError MajoranaLindblad(BdGHamiltonian(H); majorana_ops=[zeros(ComplexF64, 4)])
        @test_throws ArgumentError MajoranaLindblad(H; dephasing_ops=[(vmode,)])
    end

    @testset "BdG ground & thermal states" begin
        Random.seed!(7)
        L = 4
        hm = randn(ComplexF64, L, L); h = Hermitian((hm + hm') / 2)

        # Non-pairing: BdG ground state is the Fermi sea (fill negative-energy orbitals).
        N = count(<(0), eigvals(h))
        gs = groundstate(QuadraticHamiltonian(Matrix(h)))
        fermi = MajoranaState(CorrelationState(SlaterState(h, N)))
        @test normal_correlation(gs) ≈ normal_correlation(fermi) atol = 1e-10
        @test norm(anomalous_correlation(gs)) ≈ 0 atol = 1e-10
        @test particle_number(gs) ≈ N atol = 1e-10
        # pure: eigenvalues of iΓ are ±1
        @test all(abs.(abs.(real.(eigvals(Hermitian(im .* covariance_matrix(gs))))) .- 1) .< 1e-9)

        # quasiparticle energies = |single-particle energies| for non-pairing
        @test quasiparticle_energies(BdGHamiltonian(QuadraticHamiltonian(Matrix(h)))) ≈
              sort(abs.(eigvals(h))) atol = 1e-10

        # ground state is stationary under its own unitary evolution
        Hbdg = BdGHamiltonian(QuadraticHamiltonian(Matrix(h)))
        gevolved = evolve(gs, Hbdg, 0.83)
        @test covariance_matrix(gevolved) ≈ covariance_matrix(gs) atol = 1e-9

        # Thermal: β → ∞ recovers the ground state; non-pairing matches thermalstate(h; β)
        @test covariance_matrix(thermalstate(Hbdg; β=1e4)) ≈ covariance_matrix(gs) atol = 1e-7
        β = 0.7
        th_bdg = thermalstate(Hbdg; β=β)
        th_dirac = thermalstate(h; β=β)
        @test normal_correlation(th_bdg) ≈ correlation_matrix(th_dirac) atol = 1e-9

        # Pairing: BdG ground state has anomalous correlations, is pure and stationary.
        Bm = randn(ComplexF64, L, L); B = Bm - transpose(Bm)
        Hpair = BdGHamiltonian(Matrix(h), Matrix(B))
        gp = groundstate(Hpair)
        @test norm(anomalous_correlation(gp)) > 1e-3
        @test all(abs.(abs.(real.(eigvals(Hermitian(im .* covariance_matrix(gp))))) .- 1) .< 1e-8)
        @test covariance_matrix(evolve(gp, Hpair, 0.5)) ≈ covariance_matrix(gp) atol = 1e-8
    end

    @testset "thermalstate convention (complex H)" begin
        # Regression: thermalstate must be stationary under the package's own evolve!.
        # A complex (non-time-reversal-symmetric) H exposes the conjugation convention;
        # a real H would hide it.
        Random.seed!(3)
        hm = randn(ComplexF64, 5, 5); h = Hermitian((hm + hm') / 2)
        th = thermalstate(h; β=0.9)
        before = correlation_matrix(th)
        evolve!(th, QuadraticHamiltonian(Matrix(h)), 0.6)
        @test correlation_matrix(th) ≈ before atol = 1e-10
    end

    @testset "Convention cross-checks (vs ED)" begin
        # Validate the Majorana/Dirac conventions against explicit many-body exact
        # diagonalization for COMPLEX Hamiltonians/jumps — the regime that hid several
        # latent conjugation bugs (all prior tests used real operators).
        function fermion_ops(L)
            cop(i) = (op = ones(ComplexF64, 1, 1);
                      for k in 1:L
                          m = k < i ? ComplexF64[1 0; 0 -1] : k == i ? ComplexF64[0 1; 0 0] : ComplexF64[1 0; 0 1]
                          op = kron(m, op)
                      end; op)
            [cop(i) for i in 1:L]
        end
        Random.seed!(20); L = 3; dim = 2^L
        c = fermion_ops(L)
        x = [c[j] + c[j]' for j in 1:L]; pmaj = [im * (c[j] - c[j]') for j in 1:L]; ω = vcat(x, pmaj)
        hm = randn(ComplexF64, L, L); A = Hermitian((hm + hm') / 2)
        Bm = randn(ComplexF64, L, L); Bpair = Bm - transpose(Bm)
        Hmb = sum(A[i, j] * c[i]' * c[j] for i in 1:L, j in 1:L) +
              sum(0.5 * Bpair[i, j] * c[i]' * c[j]' + 0.5 * conj(Bpair[i, j]) * c[j] * c[i] for i in 1:L, j in 1:L)
        ψ = eigen(Hermitian((Hmb + Hmb') / 2)).vectors[:, 1]
        Γtrue = real.([(im / 2) * (ψ' * (ω[a] * ω[b] - ω[b] * ω[a]) * ψ) for a in 1:2L, b in 1:2L])
        Ctrue = [ψ' * (c[i]' * c[j]) * ψ for i in 1:L, j in 1:L]
        Ftrue = [ψ' * (c[i] * c[j]) * ψ for i in 1:L, j in 1:L]

        # fermion_correlations recovers the standard ⟨c⁺ᵢcⱼ⟩, ⟨cᵢcⱼ⟩ from the true covariance
        ms = MajoranaState(Γtrue; check=false)
        @test normal_correlation(ms) ≈ Ctrue atol = 1e-10
        @test anomalous_correlation(ms) ≈ Ftrue atol = 1e-10

        # BdG ground state matches ED in both sectors
        gp = groundstate(BdGHamiltonian(Matrix(A), Matrix(Bpair)))
        @test normal_correlation(gp) ≈ Ctrue atol = 1e-9
        @test anomalous_correlation(gp) ≈ Ftrue atol = 1e-9

        # CorrelationLindblad RHS matches ED for complex Hamiltonian + loss + gain + dephasing
        u = sqrt(0.6) .* randn(ComplexF64, L); v = sqrt(0.3) .* randn(ComplexF64, L)
        vm = normalize(randn(ComplexF64, L))
        Hop = sum(A[i, j] * c[i]' * c[j] for i in 1:L, j in 1:L)
        nd = sum(vm[i] * c[i] for i in 1:L)
        jumps = [sum(u[i] * c[i] for i in 1:L), sum(v[i] * c[i]' for i in 1:L), sqrt(0.4) * nd' * nd]
        Gr = randn(ComplexF64, dim, dim); ρ = Gr * Gr'; ρ = ρ / tr(ρ)
        C0 = [tr(ρ * c[i]' * c[j]) for i in 1:L, j in 1:L]
        Dρ = -im * (Hop * ρ - ρ * Hop) + sum(Lo * ρ * Lo' - 0.5 * (Lo' * Lo * ρ + ρ * Lo' * Lo) for Lo in jumps)
        dC_ed = [tr(Dρ * c[i]' * c[j]) for i in 1:L, j in 1:L]
        cl = CorrelationLindblad(Matrix(A); loss_ops=[u], gain_ops=[v], dephasing_ops=[(vm, 0.4)])
        @test lindblad_rhs(cl, C0) ≈ dC_ed atol = 1e-9

        # MajoranaLindblad agrees with the (now ED-correct) CorrelationLindblad for complex channels
        ml = MajoranaLindblad(QuadraticHamiltonian(Matrix(A)); loss_ops=[u], gain_ops=[v], dephasing_ops=[(vm, 0.4)])
        cs = CorrelationState(SlaterState(L=L, N=1, config="left")); mst = MajoranaState(cs)
        for dt in (0.1, 0.15); evolve!(cs, cl, dt); evolve!(mst, ml, dt); end
        @test normal_correlation(mst) ≈ correlation_matrix(cs) atol = 1e-9
    end

    @testset "Majorana trajectory (measure!)" begin
        Random.seed!(99)
        L = 4
        s = SlaterState(L, 2)
        i = 3
        qm = QuasiMode([i], ComplexF64[1], L)

        # Post-measurement covariance matches the number-conserving SlaterState collapse.
        sc1 = copy(s); GaussianFermions.apply_click!(OccupationMonitor(qm), sc1,
                                                     GaussianFermions.inner(qm, sc1.B))
        GaussianFermions.normalize!(sc1)
        m1 = MajoranaState(CorrelationState(s)); GaussianFermions._project_occupation!(m1, i, true)
        @test covariance_matrix(m1) ≈ covariance_matrix(MajoranaState(CorrelationState(sc1))) atol = 1e-9
        @test density(m1)[i] ≈ 1 atol = 1e-10

        sc0 = copy(s); GaussianFermions.apply_click!(HoleMonitor(qm), sc0,
                                                     GaussianFermions.inner(qm, sc0.B))
        GaussianFermions.normalize!(sc0)
        m0 = MajoranaState(CorrelationState(s)); GaussianFermions._project_occupation!(m0, i, false)
        @test covariance_matrix(m0) ≈ covariance_matrix(MajoranaState(CorrelationState(sc0))) atol = 1e-9
        @test density(m0)[i] ≈ 0 atol = 1e-10

        # collapsed states stay pure (eigenvalues of iΓ are ±1)
        @test all(abs.(abs.(real.(eigvals(Hermitian(im .* covariance_matrix(m1))))) .- 1) .< 1e-9)

        # Born statistics: empirical occupied-fraction ≈ ⟨nᵢ⟩
        p = density(MajoranaState(CorrelationState(s)))[i]
        rng = Random.Xoshiro(7)
        nocc = count(_ -> measure!(MajoranaState(CorrelationState(s)), i; rng), 1:4000)
        @test isapprox(nocc / 4000, p; atol=0.03)
        @test_throws ArgumentError measure!(MajoranaState([1, 0]), 5)

        # An explicit projective-monitoring trajectory loop reproduces MajoranaLindblad:
        # measuring nᵢ at rate γ ≡ dephasing D[√(2γ)nᵢ]. The caller owns the loop.
        Lc = 4; γ = 0.5; dt = 0.04; nsteps = round(Int, 1.0 / dt)
        U = propagator(BdGHamiltonian(QuadraticHamiltonian(Matrix(hopping(Lc; pbc=true).h))), dt)
        rng = Random.Xoshiro(5); ntraj = 500; acc = zeros(Lc)
        for _ in 1:ntraj
            st = MajoranaState(CorrelationState(SlaterState(L=Lc, N=2, config="Z2")))
            for _ in 1:nsteps
                evolve!(st, U)
                for site in 1:Lc
                    rand(rng) < γ * dt && measure!(st, site; rng)
                end
            end
            acc .+= density(st)
        end
        traj_density = acc ./ ntraj
        lind = MajoranaLindblad(QuadraticHamiltonian(Matrix(hopping(Lc; pbc=true).h));
                                dephasing_ops=[((v = zeros(ComplexF64, Lc); v[i] = 1; v), 2γ) for i in 1:Lc])
        mref = MajoranaState(CorrelationState(SlaterState(L=Lc, N=2, config="Z2"))); evolve!(mref, lind, 1.0)
        @test isapprox(traj_density, density(mref); atol=0.05)   # 1/√ntraj + dt error
        @test sum(traj_density) ≈ 2 atol = 0.05                  # particle number ~conserved
    end

    @testset "Majorana trajectory (linear jumps / MCWF)" begin
        # One explicit MCWF step from the primitives: draw ≤1 click over dt, else no-click.
        function mcwf_maj!(s, U, jumps, dt, rng)
            evolve!(s, U)
            rates = [jump_rate(s, ℓ) for ℓ in jumps]
            total = sum(rates)
            if rand(rng) < total * dt
                r = rand(rng) * total; idx = 1; cum = rates[1]
                while cum < r && idx < length(rates); idx += 1; cum += rates[idx]; end
                apply_click!(s, jumps[idx])
            else
                apply_noclick!(s, jumps, dt)
            end
            s
        end

        # The single-jump conditional click keeps the state physical and pure-preserving:
        # a click on a pure state stays pure (eigenvalues of iΓ remain ±1).
        ℓ = ComplexF64[0.7, 0, 0, 0, 0.3im, 0, 0, 0]            # raw linear Majorana jump
        m = MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2")))
        @test jump_rate(m, ℓ) ≥ 0                                # click rate is real, non-negative
        apply_click!(m, ℓ)
        @test all(abs.(abs.(real.(eigvals(Hermitian(im .* covariance_matrix(m))))) .- 1) .< 1e-9)

        # MCWF trajectory loop reproduces MajoranaLindblad for loss + gain (number-non-conserving).
        Lc = 4; dt = 0.01
        Hq = QuadraticHamiltonian(Matrix(hopping(Lc; pbc=true).h)); U = propagator(BdGHamiltonian(Hq), dt)
        lossw = sqrt(0.4) * (v = zeros(ComplexF64, Lc); v[1] = 1; v)
        gainw = sqrt(0.25) * (v = zeros(ComplexF64, Lc); v[3] = 1; v)
        jumps = [loss_jump(lossw), gain_jump(gainw)]
        init() = MajoranaState(CorrelationState(SlaterState(L=Lc, N=2, config="Z2")))
        lind = MajoranaLindblad(Hq; loss_ops=[lossw], gain_ops=[gainw])
        sref = init(); evolve!(sref, lind, 1.5)
        rng = Random.Xoshiro(1234); ntraj = 3000; nsteps = round(Int, 1.5 / dt); acc = zeros(Lc)
        for _ in 1:ntraj
            s = init(); for _ in 1:nsteps; mcwf_maj!(s, U, jumps, dt, rng); end; acc .+= density(s)
        end
        @test isapprox(acc ./ ntraj, density(sref); atol=0.04)

        # Pairing jump generates anomalous correlations; the MCWF loop reproduces them too.
        ℓp = sqrt(0.5) * ComplexF64[0.5, 0.5im, 0, 0, -0.5im, -0.5, 0, 0]   # L = c1 + i c2†
        @test ℓp ≈ majorana_jump(sqrt(0.5) * ComplexF64[1, 0, 0, 0], sqrt(0.5) * ComplexF64[0, 1im, 0, 0])
        srefp = init(); evolve!(srefp, MajoranaLindblad(Hq; majorana_ops=[ℓp]), 1.2)
        @test norm(anomalous_correlation(srefp)) > 0.1          # pairing is genuinely present
        rng = Random.Xoshiro(7); ntraj = 4000; nsteps = round(Int, 1.2 / dt)
        accn = zeros(Lc); accF = zeros(ComplexF64, Lc, Lc)
        for _ in 1:ntraj
            s = init(); for _ in 1:nsteps; mcwf_maj!(s, U, [ℓp], dt, rng); end
            accn .+= density(s); accF .+= anomalous_correlation(s)
        end
        @test isapprox(accn ./ ntraj, density(srefp); atol=0.04)
        @test isapprox(accF ./ ntraj, anomalous_correlation(srefp); atol=0.04)

        @test_throws ArgumentError jump_rate(m, ComplexF64[1, 0])          # wrong length
        @test_throws ArgumentError apply_click!(init(), zeros(ComplexF64, 8))  # zero-rate jump
    end

    @testset "Majorana trajectory (diffusive / QSD)" begin
        # The exact weak-measurement filter exp(α nᵢ) matches the number-conserving
        # SlaterState orbital filter (B → exp(α) on row i, renormalize) to machine precision.
        Random.seed!(11); L = 5; i = 3; α = 0.7
        s = SlaterState(L=L, N=2, config="Z2")
        s.B = Matrix(qr(randn(ComplexF64, L, L)).Q) * s.B      # generic pure state
        m = MajoranaState(CorrelationState(s))
        weak_measure!(m, i, α)
        sref = copy(s); M = Matrix{ComplexF64}(I, L, L); M[i, i] = exp(α)
        sref.B = M * sref.B; GaussianFermions.normalize!(sref)
        @test covariance_matrix(m) ≈ covariance_matrix(MajoranaState(CorrelationState(sref))) atol = 1e-12
        @test all(abs.(abs.(real.(eigvals(Hermitian(im .* covariance_matrix(m))))) .- 1) .< 1e-9)  # stays pure

        # …and matches exact diagonalization for a paired (number-non-conserving) state.
        function fermion_ops(n)
            cop(k) = (op = ones(ComplexF64, 1, 1);
                for l in 1:n
                    mm = l < k ? ComplexF64[1 0; 0 -1] : l == k ? ComplexF64[0 1; 0 0] : ComplexF64[1 0; 0 1]
                    op = kron(mm, op)
                end; op)
            [cop(k) for k in 1:n]
        end
        Random.seed!(4); n = 3
        c = fermion_ops(n); x = [c[j] + c[j]' for j in 1:n]; pm = [im * (c[j] - c[j]') for j in 1:n]
        ω = vcat(x, pm)
        hm = randn(ComplexF64, n, n); A = Hermitian((hm + hm') / 2)
        Bm = randn(ComplexF64, n, n); Bpair = Bm - transpose(Bm)
        Hmb = sum(A[a, b] * c[a]' * c[b] for a in 1:n, b in 1:n) +
              sum(0.5 * Bpair[a, b] * c[a]' * c[b]' + 0.5 * conj(Bpair[a, b]) * c[b] * c[a] for a in 1:n, b in 1:n)
        ψ = eigen(Hermitian((Hmb + Hmb') / 2)).vectors[:, 1]
        Γ0 = real.([(im / 2) * (ψ' * (ω[a] * ω[b] - ω[b] * ω[a]) * ψ) for a in 1:2n, b in 1:2n])
        β = -0.9; ψ2 = exp(β .* Matrix(c[2]' * c[2])) * ψ; ψ2 ./= norm(ψ2)
        Γed = real.([(im / 2) * (ψ2' * (ω[a] * ω[b] - ω[b] * ω[a]) * ψ2) for a in 1:2n, b in 1:2n])
        mp = MajoranaState(Γ0; check=false); weak_measure!(mp, 2, β)
        @test covariance_matrix(mp) ≈ Γed atol = 1e-10

        # Explicit QSD trajectory loop reproduces the dephasing Lindblad D[√γ nᵢ] (rate γ, not 2γ).
        Lc = 4; γ = 0.7; T = 1.0; dt = 0.01; nsteps = round(Int, T / dt)
        U = propagator(BdGHamiltonian(QuadraticHamiltonian(Matrix(hopping(Lc; pbc=true).h))), dt)
        rng = Random.Xoshiro(1); acc = zeros(Lc); ntraj = 1500
        for _ in 1:ntraj
            st = MajoranaState(CorrelationState(SlaterState(L=Lc, N=2, config="Z2")))
            for _ in 1:nsteps
                evolve!(st, U)
                for j in 1:Lc
                    α = randn(rng) * sqrt(γ * dt) + (2 * density(st, j) - 1) * γ * dt
                    weak_measure!(st, j, α)
                end
            end
            acc .+= density(st)
        end
        cl = CorrelationLindblad(Matrix(hopping(Lc; pbc=true).h);
                dephasing_ops=[((v = zeros(ComplexF64, Lc); v[j] = 1; v), γ) for j in 1:Lc])
        cs = CorrelationState(SlaterState(L=Lc, N=2, config="Z2")); evolve!(cs, cl, T)
        @test isapprox(acc ./ ntraj, real.(diag(correlation_matrix(cs))); atol=0.03)
        @test_throws ArgumentError weak_measure!(MajoranaState([1, 0]), 5, 0.1)
    end

    @testset "Channels & trajectories" begin
        rng = Random.Xoshiro(42)
        L = 8
        H = hopping(L; pbc=true)

        # explicit MCWF step from the channel primitives (no step! wrapper)
        function mcwf!(s, U, channels, dt, rng)
            evolve!(s, U)
            for ch in channels
                rate, work = jump_rate(ch, s, dt)
                if rand(rng) < rate
                    apply_click!(ch, s, work); normalize!(s)
                else
                    apply_noclick!(ch, s, dt)
                end
            end
            s
        end
        U = propagator(H, 0.05)

        # occupation monitoring conserves N
        s = SlaterState(L=L, N=4, config="Z2")
        for _ in 1:50; mcwf!(s, U, [dephasing(i, L; γ=1.0) for i in 1:L], 0.05, rng); end
        @test particle_number(s) == 4
        @test isorthonormal(s)
        @test all(-1e-9 .≤ density(s) .≤ 1 + 1e-9)

        # hole monitoring conserves N
        sh = SlaterState(L=L, N=4, config="Z2")
        for _ in 1:30; mcwf!(sh, U, [HoleMonitor(QuasiMode([i], ComplexF64[1], L); γ=1.0) for i in 1:L], 0.05, rng); end
        @test particle_number(sh) == 4
        @test isorthonormal(sh)

        # loss -> vacuum (N non-increasing)
        sl = SlaterState(L=L, N=4, config="Z2")
        for _ in 1:100; mcwf!(sl, U, [loss(i, L; γ=1.0) for i in 1:L], 0.05, rng); end
        @test particle_number(sl) == 0

        # gain -> N non-decreasing
        sg = SlaterState(L=L, N=2, config="left")
        for _ in 1:60; mcwf!(sg, U, [gain(i, L; γ=1.0) for i in 1:L], 0.05, rng); end
        @test particle_number(sg) ≥ 2
        @test isorthonormal(sg)
    end

    @testset "Continuous monitoring" begin
        rng = Random.Xoshiro(7)
        L = 8; dt = 0.05
        U = propagator(hopping(L; pbc=true), dt)
        s = SlaterState(L=L, N=4, config="Z2")
        chans = [dephasing(i, L; γ=0.5) for i in 1:L]
        for _ in 1:50
            evolve!(s, U)
            for ch in chans                                       # explicit QSD step
                qm = ch.mode
                α = randn(rng) * sqrt(ch.γ * dt) + (2 * density(s, qm) - 1) * ch.γ * dt
                weak_measure!(s, qm, α)
            end
        end
        @test particle_number(s) == 4
        @test isorthonormal(s)
    end

    @testset "No-click operators" begin
        # noclick_operator / noclick_propagator are the explicit no-jump back-action
        # builders underlying apply_noclick!/evolve_noclick! — covered directly here.
        L = 3; dt = 0.1
        cmode = QuasiMode([1, 2], ComplexF64[1, im] / sqrt(2), L)     # complex mode

        # Occupation monitor / loss: M₀ = √(1-γdt) on the mode, identity elsewhere.
        occ = OccupationMonitor(cmode; γ=0.7)
        M0 = GaussianFermions.noclick_operator(occ, dt)
        V = cmode.V; P = V * V'
        @test Matrix(M0) ≈ (sqrt(1 - 0.7 * dt) - 1) * P + I  atol = 1e-12
        # applying M₀ to the support reproduces apply_noclick! (before renormalization)
        sA = SlaterState(L=L, N=1, config="left"); sB = copy(sA)
        GaussianFermions.apply!(Matrix(M0), sA, cmode.I; normalize=false)
        apply_noclick!(occ, sB, dt; normalize=false)               # hits the same kernel
        @test sA.B ≈ sB.B  atol = 1e-12

        # Hole monitor / gain: identity on the mode, √(1-γdt) elsewhere.
        hol = HoleMonitor(cmode; γ=0.4)
        Mh = GaussianFermions.noclick_operator(hol, dt)
        a = sqrt(1 - 0.4 * dt)
        @test Matrix(Mh) ≈ (1 - a) * P + a * I  atol = 1e-12

        # noclick_propagator(G, dt) = exp(-i H_eff dt) for the effective generator.
        H = hopping(L; pbc=true)
        G = effective_hamiltonian(H, [loss(1, L; γ=0.5)])
        @test noclick_propagator(G, dt) ≈ exp(-im * dt * G.Heff)  atol = 1e-12
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

        # Feedback must wire the measurement outcome to the correct unitary. Replicate
        # its internal click / no-click branch by hand (same RNG) with nontrivial
        # complex feedback on a 2-site complex mode, and compare the resulting states.
        qm = QuasiMode([1, 2], ComplexF64[1, im] / sqrt(2), L)
        Uc = ComplexF64[0 1; 1 0]                       # nontrivial unitary on a click
        Un = ComplexF64[im 0; 0 1]                      # phase on no-click
        fb2 = Feedback(OccupationMonitor(qm; γ=0.8), Uc, Un)
        base = SlaterState(L=L, N=2, config="Z2")
        sfb = copy(base); sman = copy(base)
        clicked = apply!(fb2, sfb, 0.3; rng=Random.Xoshiro(123))
        let ch = fb2.channel, rman = Random.Xoshiro(123)
            rate, v = jump_rate(ch, sman, 0.3)
            if rand(rman) < rate
                apply_click!(ch, sman, v); normalize!(sman)
                GaussianFermions.apply!(Uc, sman, ch.mode.I; threads=true)
                @test clicked == true
            else
                apply_noclick!(ch, sman, 0.3; normalize=true)
                GaussianFermions.apply!(Un, sman, ch.mode.I; threads=true)
                @test clicked == false
            end
        end
        @test sfb.B ≈ sman.B  atol = 1e-12
        @test isorthonormal(sfb)
    end

    @testset "Trajectory averaging (manual loop)" begin
        # The library has no ensemble runner; the caller owns the averaging loop.
        rng = Random.Xoshiro(11)
        L = 8; dt = 0.1; nsteps = 10; ntraj = 24
        U = propagator(hopping(L; pbc=true), dt)
        chans = [dephasing(i, L; γ=0.5) for i in 1:L]
        accn = zeros(L)
        for _ in 1:ntraj
            s = SlaterState(L=L, N=4, config="Z2")
            for _ in 1:nsteps
                evolve!(s, U)
                for ch in chans
                    rate, work = jump_rate(ch, s, dt)
                    rand(rng) < rate ? (apply_click!(ch, s, work); normalize!(s)) : apply_noclick!(ch, s, dt)
                end
            end
            accn .+= density(s)
        end
        meann = accn ./ ntraj
        @test all(0 .≤ meann .≤ 1)
        @test sum(meann) ≈ 4 atol = 1e-6                 # occupation monitoring conserves N
    end

    @testset "Examples" begin
        # Smoke-test the shipped examples at tiny sizes. The `PROGRAM_FILE` guard in
        # each file skips its heavy driver on `include`; we call the light entry
        # points directly. This guards against API drift silently breaking examples.
        exdir = joinpath(@__DIR__, "..", "example")
        include(joinpath(exdir, "FreeFermion.jl"))
        include(joinpath(exdir, "MI.jl"))
        include(joinpath(exdir, "Dephase.jl"))

        @test freefermion_demo(; L=6, ntraj=2, tspan=0.2, dt=0.05).ntraj == 2
        @test mi_vs_gamma(; L=8, γ=0.5, ntraj=2, tspan=0.5, dt=0.05) isa Real

        ref = lindblad_den_evo(; L=8, d=[1, 3, 1], T=0.5)
        maj = majorana_den_evo(; L=8, d=[1, 3, 1], T=0.5)
        jmp = jump_den_evo(; L=8, d=[1, 3, 1], ntraj=4, T=0.5)
        qsd = qsd_den_evo(; L=8, d=[1, 3, 1], ntraj=4, T=0.5)
        @test size(ref) == size(maj) == size(jmp) == size(qsd) == (8, 10)
        @test all(0 .≤ ref .≤ 1)
        @test maj ≈ ref atol = 1e-9   # Dirac and Majorana master equations agree
    end

end
