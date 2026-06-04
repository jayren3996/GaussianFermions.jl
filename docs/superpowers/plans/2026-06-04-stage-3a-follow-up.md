# Stage 3A Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small public API and documentation follow-up to Stage 3A.

**Architecture:** Keep Stage 3A intact. Extend existing BdG spectral helpers in
`src/BdGHamiltonians.jl`, extend finite constructors in `src/Models.jl`, and keep
all existing names as compatibility wrappers. Update docs in place.

**Tech Stack:** Julia, LinearAlgebra dense eigensolvers, existing Test suite,
Documenter docs.

---

## File Structure

- Modify `src/BdGHamiltonians.jl`: spectrum names and zero-mode ground-state policy.
- Modify `src/Models.jl`: constructor `flux` keyword and BdG energy wrapper name.
- Modify `test/runtests.jl`: regression tests for the new API and constructor flux.
- Modify docs and README files listed in the design spec.
- Modify `docs/superpowers/specs/2026-06-04-stage-3-roadmap-notes.md`: next slice wording.

## Task 1: BdG Spectrum Names And Zero-Mode Policy

- [ ] Write failing tests for `nambu_spectrum`, `quasiparticle_spectrum`,
      compatibility wrappers, and `groundstate(...; zero_mode=...)`.
- [ ] Run `julia --project=. test/runtests.jl` and confirm failures are for missing
      API / current zero-mode behavior.
- [ ] Export and implement `nambu_spectrum` and `quasiparticle_spectrum`.
- [ ] Implement `groundstate(H::BdGHamiltonian; zero_mode=:error, atol=1e-10)`.
- [ ] Forward `groundstate(H::QuadraticHamiltonian; kwargs...)`.
- [ ] Run `julia --project=. test/runtests.jl` and confirm the new tests pass.
- [ ] Commit: `Add explicit BdG spectrum and zero-mode APIs`.

## Task 2: Constructor Boundary Flux

- [ ] Write failing tests for periodic flux on SSH, Aubry-Andre, and Kitaev closing
      bonds, plus open-chain flux rejection.
- [ ] Run `julia --project=. test/runtests.jl` and confirm expected failures.
- [ ] Add `flux::Real=0.0` to the three finite constructors.
- [ ] Apply `exp(im * flux)` to periodic closing hopping/pairing bonds.
- [ ] Run `julia --project=. test/runtests.jl`.
- [ ] Commit: `Add boundary flux to finite constructors`.

## Task 3: Front-Door Docs And Roadmap

- [ ] Update README and docs homepage to mention model/spectral helpers and fix
      "two state representations" to "three state representations".
- [ ] Update reference/manual/example docs for `quasiparticle_spectrum`,
      `nambu_spectrum`, and the zero-mode policy.
- [ ] Update Stage 3 roadmap notes to make Wick/Pfaffian observables the next
      feature slice.
- [ ] Run `julia --project=docs docs/make.jl`.
- [ ] Commit: `Document Stage 3A follow-up APIs`.

## Final Verification

- [ ] Run `julia --project=. test/runtests.jl`.
- [ ] Run `julia --project=docs docs/make.jl`.
- [ ] Run `git diff --check`.
- [ ] Request code review before publishing or merging.

