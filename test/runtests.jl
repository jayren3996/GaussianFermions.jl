using GaussianFermions
using Test, LinearAlgebra, Random

# Helper: are the columns of a FreeFermionState orthonormal? (physical pure state)
isorthonormal(s) = isapprox(s.B' * s.B, I(rank(s)); atol=1e-10)

@testset "GaussianFermions.jl" begin

    @testset "Construction & correlations" begin
        s = FreeFermionState([1, 3], 4)
        @test rank(s) == 2
        @test diag(s) ≈ [1, 0, 1, 0]
        @test isorthonormal(s)
        # correlation matrix of a product state is diagonal = occupation
        @test s[:] ≈ diagm([1, 0, 1, 0])
        @test s[1, 1] ≈ 1
        @test s[2, 2] ≈ 0

        # Boolean-vector constructor agrees with position constructor
        sb = FreeFermionState([true, false, true, false])
        @test sb.B == s.B

        # config presets
        @test diag(FreeFermionState(L=4, N=2, config="left"))   ≈ [1, 1, 0, 0]
        @test diag(FreeFermionState(L=4, N=2, config="right"))  ≈ [0, 0, 1, 1]
        @test diag(FreeFermionState(L=4, N=2, config="center")) ≈ [0, 1, 1, 0]
        @test diag(FreeFermionState(L=4, N=2, config="Z2"))     ≈ [1, 0, 1, 0]
        # "random" must not throw (regression: missing Random import) and give N particles
        Random.seed!(1)
        sr = FreeFermionState(L=8, N=3, config="random")
        @test rank(sr) == 3
        @test sum(diag(sr)) ≈ 3
    end

    @testset "Entanglement entropy" begin
        # product state has zero entanglement
        s = FreeFermionState([1, 3], 4)
        @test ent_S(s, [1, 2]) ≈ 0 atol=1e-10
        # mutual information of disjoint product regions is zero
        @test ent_S(s, [1], [3]) ≈ 0 atol=1e-10

        # one particle in a 50/50 superposition over two sites -> S = log 2
        s2 = FreeFermionState([1], 2)
        U = ComplexF64[1 1; 1 -1] / sqrt(2)
        apply!(U, s2)
        @test ent_S(s2, [1]) ≈ log(2) atol=1e-8
    end

    @testset "Unitary evolution" begin
        L = 6
        H = diagm(1 => ones(L-1), -1 => ones(L-1))
        U = evo_operator(Hermitian(H), 0.3)
        @test U * U' ≈ I(L)                         # unitarity
        s = FreeFermionState(L=L, N=3, config="Z2")
        n0 = sum(diag(s))
        s = U * s
        @test sum(diag(s)) ≈ n0                     # particle number conserved
        @test isorthonormal(s)
        @test rank(s) == 3

        # local apply! (turbo path) matches full-matrix application
        sA = FreeFermionState([1], 2); sB = FreeFermionState([1], 2)
        g = ComplexF64[1 1; 1 -1] / sqrt(2)
        apply!(g, sA)              # full matrix
        apply!(g, sB, [1, 2])      # local kernel
        @test sA.B ≈ sB.B
    end

    @testset "Projective measurement" begin
        # measuring an occupied mode yields outcome `true` deterministically (⟨n⟩=1)
        s = FreeFermionState([1, 3], 4)
        @test measure!(QuasiMode([1], ComplexF64[1], 4), s) == true
        # measuring an empty mode yields `false` (⟨n⟩=0); site stays empty
        s2 = FreeFermionState([1, 3], 4)
        @test measure!(QuasiMode([2], ComplexF64[1], 4), s2) == false
        @test isorthonormal(s2)
    end

    @testset "Quantum-jump trajectories" begin
        Random.seed!(42)
        L = 8
        U = evo_operator(Hermitian(diagm(1 => ones(L-1), -1 => ones(L-1))), 0.1)

        # number-conserving monitoring (QJParticle): N preserved, state stays physical
        jp = [QJParticle(QuasiMode([i], ComplexF64[1], L), 0.3) for i in 1:L]
        s = FreeFermionState(L=L, N=4, config="Z2")
        for _ in 1:50
            apply!(U, s); apply!(jp, s)
        end
        @test rank(s) == 4
        @test isorthonormal(s)
        @test all(0 - 1e-9 .≤ diag(s) .≤ 1 + 1e-9)

        # loss channel (QJDrain): particle number can only decrease
        sd = FreeFermionState(L=L, N=4, config="Z2")
        jd = [QJDrain(QuasiMode([i], ComplexF64[1], L), 0.5) for i in 1:L]
        for _ in 1:50
            apply!(U, sd); apply!(jd, sd)
        end
        @test rank(sd) ≤ 4
        @test isorthonormal(sd)

        # hole channel (QJHole): regression for the avoid_vector! keyword bug
        sh = FreeFermionState(L=L, N=4, config="Z2")
        jh = [QJHole(QuasiMode([i], ComplexF64[1], L), 0.5) for i in 1:L]
        for _ in 1:20
            apply!(U, sh); apply!(jh, sh)
        end
        @test rank(sh) == 4
        @test isorthonormal(sh)
    end

    @testset "Continuous monitoring (wiener!)" begin
        Random.seed!(7)
        L = 8
        U = evo_operator(Hermitian(diagm(1 => ones(L-1), -1 => ones(L-1))), 0.1)
        qms = [QuasiMode([i], ComplexF64[1], L) for i in 1:L]
        s = FreeFermionState(L=L, N=4, config="Z2")
        for _ in 1:50
            apply!(U, s); wiener!(qms, s, 0.2)
        end
        @test rank(s) == 4
        @test isorthonormal(s)
        @test all(0 - 1e-9 .≤ diag(s) .≤ 1 + 1e-9)
    end

    @testset "Conditional feedback" begin
        Random.seed!(3)
        L = 4
        # ConditionalJump: jump on a single site with trivial (identity) feedback
        s = FreeFermionState(L=L, N=2, config="Z2")
        cj = ConditionalJump(QJParticle(QuasiMode([1], ComplexF64[1], L), 0.5),
                             ComplexF64[1;;])
        @test apply!(cj, s) isa Bool
        @test isorthonormal(s)

        # ConditionalMeasure: regression for the `measure`/`cj` typos
        s2 = FreeFermionState(L=L, N=2, config="Z2")
        cm = ConditionalMeasure(QuasiMode([1], ComplexF64[1], L),
                                ComplexF64[1;;], ComplexF64[1;;])
        @test apply!(cm, s2) isa Bool
        @test isorthonormal(s2)
    end

    @testset "Covariance / Majorana formalism" begin
        # covariance matrix -> Dirac correlation recovers the occupation
        Γ = covariancematrix([1, 0, 1, 0])
        A, _ = fermioncorrelation(Γ)
        @test real(diag(A)) ≈ [1, 0, 1, 0] atol=1e-12
        @test fermioncorrelation(Γ, 1) ≈ A

        # majoranaform of a Hermitian hopping matrix is real & antisymmetric
        H = diagm(1 => ones(3), -1 => ones(3))
        Hm = majoranaform(H)
        @test Hm ≈ -transpose(Hm)
        @test eltype(Hm) <: Real

        # von Neumann entropy of a pure (product) state is zero (regression: `logb`)
        @test GaussianFermions.entropy(Γ) ≈ 0 atol=1e-10
        @test GaussianFermions.entropy(Γ, [1, 2]) ≈ 0 atol=1e-10
    end

    @testset "Lindblad evolution" begin
        L = 4
        H = diagm(1 => ones(L-1), -1 => ones(L-1))
        nd = normalize([1.0, 1.0])
        I_ind = [i:i+1 for i in 1:2:L-1]
        M = fill(sqrt(0.5) * nd * nd', length(I_ind))

        # Majorana-covariance Lindbladian
        lind = quadraticlindblad_from_fermion(; H=H, M=M, I=I_ind)
        Γ0 = covariancematrix([1, 0, 1, 0])
        sol = lindblad_evo(lind, Γ0, [0.0, 0.5, 1.0])
        Γend = sol[end]
        @test all(isfinite, Γend)
        @test Γend ≈ -transpose(Γend) atol=1e-6     # stays antisymmetric
        nden = real(diag(fermioncorrelation(Γend, 1)))
        @test all(-1e-6 .≤ nden .≤ 1 + 1e-6)

        # Dirac correlation-matrix Lindbladian (regression for the aliasing fix)
        fl = fermionlindblad(H, M, I_ind)
        G0 = diagm(ComplexF64[1, 0, 1, 0])
        solf = lindblad_evo(fl, G0, [0.0, 0.5, 1.0])
        @test all(isfinite, solf[end])
        @test all(-1e-6 .≤ real(diag(solf[end])) .≤ 1 + 1e-6)
    end

end
