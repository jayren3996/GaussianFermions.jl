# Documentation Depth Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a logo hero + branding, revive Getting Started, drop two stale pages, deepen the manual and all examples, and add monitored-dynamics / measurement-induced entanglement transition examples to the Documenter.jl site.

**Architecture:** Documenter.jl static site under `docs/`. Pages are Markdown in `docs/src`; deterministic snippets are `@example` blocks executed by the build, stochastic snippets are fixed-seed plain `julia` blocks. Nav is declared in `docs/make.jl`. The build is the integration test: `julia --project=docs docs/make.jl` must finish with no errored examples, broken links, or missing-docstring errors.

**Tech Stack:** Julia 1.10+ (1.12 locally), Documenter.jl v1, GaussianFermions.jl (dev-ed into `docs/Project.toml`).

**Conventions baked in (verified against source + a live run):**
- `BdGHamiltonian(A, B)`: `A` Hermitian, `B` antisymmetric; `H = Σ Aᵢⱼc⁺ᵢcⱼ + ½Σ(Bᵢⱼc⁺ᵢc⁺ⱼ + h.c.)`.
- Channel `jump_rate(ch, s, dt)` returns `(probability, work)`; Majorana `jump_rate(s, ℓ)` returns an instantaneous **rate**.
- `parity` is defined for number-conserving states only — **do NOT call it on a `MajoranaState`** (MethodError).
- `groundstate(BdGHamiltonian)` fails at an **exact** zero mode (parity-degenerate); evaluate it at a slightly detuned point.
- Projective occupation monitoring of `nᵢ` at rate `γ` unravels `D[√(2γ) nᵢ]`; `OccupationMonitor(γ)` and QSD `weak_measure!` (rate `γ`) unravel `D[√γ nᵢ]`. To compare all three against one Lindbladian `D[√γ nᵢ]`, use channel/QSD rate `γ` and projective rate `γ/2`.

---

## File Structure

**Create:**
- `docs/src/examples/kitaev-chain.md` — BdG/topology showcase (deterministic).
- `docs/src/examples/measurement-induced-transition.md` — monitored entanglement transition (stochastic).
- `docs/src/examples/monitoring-protocols.md` — projective/MCWF/QSD comparison (stochastic).
- `example/Transition.jl` — full-scale transition run (backs the transition page).
- `example/Protocols.jl` — full-scale protocol comparison (backs the protocols page).
- (Maybe) `docs/src/assets/logo-dark.svg` — only if the build shows the logo is illegible on dark.

**Modify:**
- `docs/make.jl` — add Getting Started, drop the two stale pages, reorder/extend Examples nav.
- `docs/src/index.md` — logo hero.
- `docs/src/getting-started.md` — rewrite with `@example` + fixed links + trajectory teaser.
- `docs/src/manual/{conventions,states,hamiltonians,lindblad,trajectories,observables}.md` — admonitions, `@example` conversion, cross-links.
- `docs/src/examples/{free-fermion-chain,deterministic-lindblad,trajectories-vs-lindblad,bdg-pairing,monitored-mutual-information}.md` — deepen.
- `README.md` — add the two new `example/*.jl` scripts to the Examples list.

**Delete:**
- `docs/src/manual/majorana-bdg.md`, `docs/src/manual/correlation-lindblad.md`.

---

## Phase 0 — Baseline

### Task 1: Confirm a clean baseline build

**Files:** none (verification only).

- [ ] **Step 1: Build the site from the current tree**

Run:
```bash
julia --project=docs docs/make.jl
```
Expected: ends with `[ Info: Automatic `version="..."`` / no `Error:` lines; exit code 0. If `@example` blocks already fail on this machine, stop and report — the baseline must be green before changes.

- [ ] **Step 2: Note the logo/favicon wiring**

Run:
```bash
ls docs/build/assets/logo.svg docs/build/assets/logo.png docs/build/assets/favicon.ico 2>&1
grep -o 'logo[^"]*' docs/build/index.html | head
```
Expected: the assets exist in `build/assets`; `index.html` references the sidebar logo. This confirms Documenter's auto-detection works (no `make.jl` change needed for the sidebar logo/favicon).

No commit (baseline only; `docs/build` is gitignored).

---

## Phase 1 — Branding & Structure

### Task 2: Logo hero on the home page

**Files:**
- Modify: `docs/src/index.md:1`

- [ ] **Step 1: Prepend a centered logo hero**

Insert at the very top of `docs/src/index.md`, before the current `# GaussianFermions.jl` line:

````markdown
```@raw html
<div align="center">
<img src="assets/logo.svg" alt="GaussianFermions.jl logo" width="180"/>
</div>
```

````
Leave the existing `# GaussianFermions.jl` H1 and the rest of the page unchanged. (The `assets/logo.svg` path is relative to the built `index.html`, which sits beside `assets/`.)

- [ ] **Step 2: Rebuild and confirm the hero renders, logo legible on dark**

Run:
```bash
julia --project=docs docs/make.jl
grep -c 'assets/logo.svg' docs/build/index.html
```
Expected: build succeeds; count ≥ 1. Open `docs/build/index.html` and toggle the dark theme (the moon icon); confirm the sidebar logo and hero are legible. The SVG uses bright teal/violet gradients + Julia-hue dots on transparent bg (designed for both themes), so **no `logo-dark.svg` is expected**. Only if the build/inspection shows it washing out, add `docs/src/assets/logo-dark.svg` (a copy with darker strokes) and rebuild.

- [ ] **Step 3: Commit**

```bash
git add docs/src/index.md
git commit -m "docs: add logo hero to docs home page"
```

### Task 3: Revive Getting Started and wire it into the nav

**Files:**
- Modify: `docs/src/getting-started.md` (full rewrite)
- Modify: `docs/make.jl:21`

- [ ] **Step 1: Rewrite `docs/src/getting-started.md`**

Replace the entire file with (deterministic snippets are `@example getting_started`, links point at live pages, and a trajectory teaser forwards to the manual):

````markdown
# Getting Started

This page walks through one full session: prepare a state, build a Hamiltonian,
evolve it, measure observables, and add dissipation. Every code block runs as part
of the documentation build.

## Installation

```julia
pkg> add https://github.com/jayren3996/GaussianFermions.jl
```

```julia
using GaussianFermions
```

## 1. Prepare a state

A pure number-conserving free-fermion state is a [`SlaterState`](@ref). Here a
half-filled chain in a `Z2` (alternating-occupation) product configuration:

```@example getting_started
using GaussianFermions

L = 8
s = SlaterState(L=L, N=4, config="Z2")
density(s)
```

For correlation-matrix dynamics, wrap it in a [`CorrelationState`](@ref):

```@example getting_started
state = CorrelationState(s)
particle_number(state)
```

## 2. Build a Hamiltonian

Free-fermion (single-particle) Hamiltonians are `QuadraticHamiltonian`s, with
convenience builders for the common chain terms:

```@example getting_started
H = hopping(L; pbc=true)          # nearest-neighbour hopping, periodic
nothing # hide
```

## 3. Evolve

`evolve!` advances a state in place; for repeated fixed steps, precompute a
`propagator`:

```@example getting_started
evolve!(state, H, 0.5)            # evolve for time t = 0.5
U = propagator(H, 0.05)
for _ in 1:20
    evolve!(state, U)
end
round(particle_number(state); digits=8)
```

## 4. Measure observables

```@example getting_started
(density        = round.(density(state); digits=3),
 number         = round(particle_number(state); digits=6),
 half_chain_S   = round(entanglement_entropy(state, 1:4); digits=4))
```

## 5. Add dissipation

Number-conserving open dynamics use `CorrelationLindblad`. Loss/gain inputs are
amplitude vectors: for rate `γ` on a normalized mode `v`, pass `√γ · v`.

```@example getting_started
loss1 = zeros(ComplexF64, L); loss1[1] = sqrt(0.2)   # loss at site 1, rate 0.2
lind  = CorrelationLindblad(H; loss_ops=[loss1])

open_state = CorrelationState(SlaterState(L=L, N=4, config="Z2"))
evolve!(open_state, lind, 1.0)
round.(density(open_state); digits=3)
```

## 6. Monitor a trajectory

There is no ensemble runner: you own the loop. A single quantum-jump (MCWF) step
evolves, then applies per-channel click / no-click back-action:

```@example getting_started
using Random

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

rng = Xoshiro(1)
Um  = propagator(hopping(L; pbc=true), 0.1)
chans = [dephasing(i, L; γ=0.5) for i in 1:L]
traj = SlaterState(L=L, N=4, config="Z2")
for _ in 1:50
    mcwf_step!(traj, Um, chans, 0.1; rng)
end
round(entanglement_entropy(traj, 1:4); digits=4)
```

## Next steps

- [Conventions](manual/conventions.md) — read before comparing signs or correlations
  with another codebase.
- [States](manual/states.md) and [Hamiltonians & Time Evolution](manual/hamiltonians.md)
  — model a closed system.
- [Lindblad Dynamics](manual/lindblad.md) — deterministic open-system evolution.
- [Quantum Trajectories](manual/trajectories.md) — monitored dynamics, plus the
  [Measurement-Induced Transition](examples/measurement-induced-transition.md) example.
- [API Reference](reference/overview.md) — signatures and mutation behavior.
````

- [ ] **Step 2: Add it to the nav**

In `docs/make.jl`, change the line:
```julia
        "Home" => "index.md",
```
to:
```julia
        "Home" => "index.md",
        "Getting Started" => "getting-started.md",
```

- [ ] **Step 3: Build and verify**

Run:
```bash
julia --project=docs docs/make.jl
```
Expected: build succeeds, `getting_started` examples run, no broken `@ref`/links. Confirm `docs/build/getting-started.html` exists and is in the sidebar nav.

- [ ] **Step 4: Commit**

```bash
git add docs/src/getting-started.md docs/make.jl
git commit -m "docs: revive Getting Started page with runnable walkthrough"
```

### Task 4: Remove the two stale manual pages

**Files:**
- Delete: `docs/src/manual/majorana-bdg.md`, `docs/src/manual/correlation-lindblad.md`

- [ ] **Step 1: Confirm coverage in the surviving pages**

Run:
```bash
grep -l "groundstate\|quasiparticle_energies\|thermalstate" docs/src/manual/hamiltonians.md
grep -l "CorrelationLindblad\|MajoranaLindblad\|steadystate\|lindblad_rhs" docs/src/manual/lindblad.md
grep -l "MajoranaState\|anomalous_correlation" docs/src/manual/states.md
```
Expected: each prints its filename (content is covered). The stale pages add nothing the survivors lack.

- [ ] **Step 2: Delete the files and fix inbound links**

```bash
git rm docs/src/manual/majorana-bdg.md docs/src/manual/correlation-lindblad.md
grep -rn "majorana-bdg\|correlation-lindblad" docs/src docs/make.jl
```
Expected after deletion: the second grep returns nothing (the only referrers were the now-rewritten `getting-started.md` and the two deleted files themselves). If any reference remains, repoint it to `manual/lindblad.md` (Lindblad topics) or `manual/hamiltonians.md` (BdG/ground/thermal topics).

- [ ] **Step 3: Build and verify no broken links**

Run:
```bash
julia --project=docs docs/make.jl 2>&1 | grep -i "warn\|error\|broken" || echo "clean"
```
Expected: `clean` (or only pre-existing unrelated warnings). The two pages are not in the nav, so removing them cannot orphan nav entries.

- [ ] **Step 4: Commit**

```bash
git add -A docs/src/manual docs/make.jl
git commit -m "docs: remove stale majorana-bdg and correlation-lindblad pages"
```

---

## Phase 2 — Manual Deepening

### Task 5: Manual consistency pass (admonitions, `@example`, cross-links)

**Files:**
- Modify: `docs/src/manual/conventions.md`, `states.md`, `hamiltonians.md`, `lindblad.md`, `observables.md`

- [ ] **Step 1: `conventions.md` — add a mutation/rate warning**

Append to the end of `docs/src/manual/conventions.md`:
````markdown

!!! warning "Rates vs probabilities"
    The channel form [`jump_rate(ch, s, dt)`](@ref) returns a click *probability*
    over the step `dt` (and a reusable work vector). The Majorana form
    [`jump_rate(s, ℓ)`](@ref) returns an instantaneous *rate*; multiply by `dt`
    yourself when drawing events.

!!! note "Mutation"
    Functions ending in `!` mutate their state argument (`evolve!`, `measure!`,
    `weak_measure!`, `apply_click!`, `apply_noclick!`). The non-bang `evolve`
    returns an evolved copy.
````

- [ ] **Step 2: `states.md` — cross-link to examples**

Append to the end of `docs/src/manual/states.md`:
````markdown

See [Free-Fermion Chain](../examples/free-fermion-chain.md) for `SlaterState`
workflows, [BdG Pairing](../examples/bdg-pairing.md) and
[Kitaev Chain](../examples/kitaev-chain.md) for `MajoranaState` workflows, and the
[States & Modes API reference](../reference/states.md) for signatures.
````

- [ ] **Step 3: `hamiltonians.md` — make the BdG ground/thermal block runnable**

In `docs/src/manual/hamiltonians.md`, change the block currently fenced as
```` ```julia ```` containing `gs = groundstate(bdg)` to a runnable example:
````markdown
```@example hamiltonians
gs  = groundstate(bdg)
eps = quasiparticle_energies(bdg)
rho = thermalstate(bdg; β=2.0)
(min_energy = round(minimum(eps); digits=4),
 gs_number  = round(particle_number(gs); digits=4))
```
````
Leave the `Htotal = H + mu` and `bdg_hopping = ...` blocks as plain `julia` (they show construction without output worth checking).

- [ ] **Step 4: `lindblad.md` — add the unraveling note**

In `docs/src/manual/lindblad.md`, under `## Relation To Trajectories`, append:
````markdown

!!! note "Linear vs nonlinear observables"
    Different unravelings of the same Lindbladian agree on **linear** observables
    (e.g. density) after averaging, but **entanglement is trajectory-nonlinear** and
    depends on the unraveling — it must be computed as a trajectory average, never
    from the deterministic solver. See
    [Monitoring Protocols](../examples/monitoring-protocols.md) and
    [Measurement-Induced Transition](../examples/measurement-induced-transition.md).
````

- [ ] **Step 5: `observables.md` — make the information-measures block runnable + add the note**

In `docs/src/manual/observables.md`, replace the `## Information Measures` plain
`julia` block with a runnable one and a note:
````markdown
## Information Measures

Multipartite diagnostics are central to monitored dynamics:

```@example observables
A = 1:2; B = 3:4; C = 5:6
(mutual     = round(mutual_information(s, A, B); digits=4),
 tripartite = round(tripartite_information(s, A, B, C); digits=4))
```

!!! note
    `mutual_information` and `tripartite_information` are nonlinear in the state, so
    in trajectory studies the caller computes them inside the sampling loop and
    averages over trajectories or a late-time window. `tripartite_information` is the
    scale-invariant order parameter used in the
    [Measurement-Induced Transition](../examples/measurement-induced-transition.md)
    and [Mutual & Tripartite Information](../examples/monitored-mutual-information.md)
    examples.
````

- [ ] **Step 6: Build and verify**

Run:
```bash
julia --project=docs docs/make.jl
```
Expected: build succeeds; the new `@example hamiltonians` and `@example observables` blocks execute (they reference `bdg` and `s` already defined earlier on their pages — confirm those names exist on the page; if not, define them in the block).

- [ ] **Step 7: Commit**

```bash
git add docs/src/manual
git commit -m "docs: add admonitions, runnable blocks, and cross-links to the manual"
```

### Task 6: Deepen `trajectories.md`

**Files:**
- Modify: `docs/src/manual/trajectories.md`

- [ ] **Step 1: Convert the deterministic Majorana-jump setup to runnable, add a rate warning**

In `docs/src/manual/trajectories.md`, under `## Majorana Linear Jumps`, insert a
`!!! warning` right after the heading:
````markdown
!!! warning
    `jump_rate(s::MajoranaState, ℓ)` returns an instantaneous **rate** `⟨L†L⟩`, not a
    probability. The per-step click probability is `jump_rate(s, ℓ) * dt`. This
    differs from the channel form `jump_rate(ch, s, dt)`, which already includes `dt`.
````
Make the jump-vector construction block runnable:
````markdown
```@example trajectories
using GaussianFermions

s = MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2")))
jumps = [
    loss_jump(sqrt(0.4) * ComplexF64[1, 0, 0, 0]),
    gain_jump(sqrt(0.25) * ComplexF64[0, 0, 1, 0]),
]
round.([jump_rate(s, ℓ) for ℓ in jumps]; digits=4)
```
````
Keep the projective/weak-measurement snippets that draw randomness as plain `julia`
blocks, but give each a fixed seed (`rng = Xoshiro(1)`) so they are reproducible if a
reader runs them.

- [ ] **Step 2: Build and verify**

Run:
```bash
julia --project=docs docs/make.jl
```
Expected: build succeeds; `@example trajectories` runs and prints two rates.

- [ ] **Step 3: Commit**

```bash
git add docs/src/manual/trajectories.md
git commit -m "docs: deepen trajectories manual page with runnable rates and rate warning"
```

---

## Phase 3 — Examples

> For every example task: after writing the page, run a fast snippet check, then a
> full build at the end of the phase. The snippet check command is given per task.

### Task 7: Deepen `free-fermion-chain.md`

**Files:**
- Modify: `docs/src/examples/free-fermion-chain.md` (full rewrite)

- [ ] **Step 1: Replace the file**

````markdown
# Free-Fermion Chain

This workflow shows the number-conserving path end to end: prepare a Slater
determinant, evolve under a hopping Hamiltonian, then run a full monitored
trajectory and average an observable over the ensemble — the loop you own.

## Closed evolution

```@example freefermion
using GaussianFermions

L = 12
H = hopping(L; pbc=true)
s = SlaterState(L=L, N=L ÷ 2, config="Z2")

S0 = entanglement_entropy(s, 1:(L ÷ 2))
evolve!(s, H, 1.0)
S1 = entanglement_entropy(s, 1:(L ÷ 2))

(initial_entropy=round(S0; digits=4),
 final_entropy=round(S1; digits=4),
 particle_number=particle_number(s))
```

For fixed time steps, precompute the propagator once:

```@example freefermion
U = propagator(H, 0.05)
for _ in 1:100
    evolve!(s, U)
end
round(particle_number(s); digits=8)   # conserved
```

## A monitored trajectory

There is no ensemble runner. One MCWF step evolves, then applies per-channel
click / no-click back-action:

```julia
using GaussianFermions, Random

function mcwf_step!(s, U, channels, dt; rng=Random.default_rng())
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
```

## Trajectory average

The caller owns the averaging loop too. Here, the late-time half-chain
entanglement under occupation monitoring, with a standard error from the
ensemble spread:

```julia
function ensemble_entropy(; L=16, γ=0.5, dt=0.05, tspan=5.0, ntraj=64, rng=Xoshiro(1))
    U     = propagator(hopping(L; pbc=true), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    nsteps = round(Int, tspan / dt)
    ΣS = 0.0; ΣS² = 0.0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for _ in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
        end
        S = entanglement_entropy(s, 1:(L ÷ 2))
        ΣS += S; ΣS² += S^2
    end
    mean = ΣS / ntraj
    stderr = sqrt(max(ΣS² / ntraj - mean^2, 0.0)) / sqrt(ntraj)
    (mean=round(mean; digits=4), stderr=round(stderr; digits=4))
end

ensemble_entropy()
# (mean ≈ 1.5–2.0, stderr ≈ 0.05 for these small-ensemble parameters)
```

The full script [`example/FreeFermion.jl`](https://github.com/jayren3996/GaussianFermions.jl/blob/main/example/FreeFermion.jl)
runs this at larger sizes. For the monitoring-strength dependence of the
steady-state entanglement, see the
[Measurement-Induced Transition](measurement-induced-transition.md) example.
````

- [ ] **Step 2: Snippet check**

Run:
```bash
julia --project=docs -e 'using GaussianFermions; L=12; H=hopping(L;pbc=true); s=SlaterState(L=L,N=L÷2,config="Z2"); evolve!(s,H,1.0); println(round(entanglement_entropy(s,1:6);digits=4), " ", particle_number(s))'
```
Expected: prints an entropy and `6` (particle number conserved).

- [ ] **Step 3: Commit**

```bash
git add docs/src/examples/free-fermion-chain.md
git commit -m "docs: deepen free-fermion-chain example with full trajectory loop"
```

### Task 8: Deepen `deterministic-lindblad.md`

**Files:**
- Modify: `docs/src/examples/deterministic-lindblad.md` (full rewrite)

- [ ] **Step 1: Replace the file** (absorbs the deleted `correlation-lindblad.md` content)

````markdown
# Deterministic Lindblad

`CorrelationLindblad` evolves the number-conserving correlation matrix `C` under a
quadratic Lindblad master equation. It is dense and intended for finite systems. Loss
and gain inputs are amplitude vectors: rate `γ` on a normalized mode `v` is `√γ · v`.

## Local particle loss

```@example deterministic_lindblad
using GaussianFermions

L = 10
H = hopping(L; pbc=false)

loss_left = zeros(ComplexF64, L)
loss_left[1] = sqrt(0.4)

lind  = CorrelationLindblad(H; loss_ops=[loss_left])
state = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))

for _ in 1:20
    evolve!(state, lind, 0.05)
end
round.(density(state); digits=3)
```

## From trajectory channels

The same constructor accepts the number-conserving channel objects from the
trajectory API, which is convenient when comparing a master equation with its
unravelings:

```@example deterministic_lindblad
channels = [dephasing(i, L; γ=0.5) for i in 1:L]
lind2 = CorrelationLindblad(H, channels)
nothing # hide
```

The right-hand side is exposed as `lindblad_rhs` for use in a custom integrator.

## Steady states

For a finite system with a unique attractive steady state, solve the affine problem
directly instead of integrating to long times:

```@example deterministic_lindblad
steady = steadystate(CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                         loss_ops=[ComplexF64[sqrt(0.6)]],
                                         gain_ops=[ComplexF64[sqrt(0.2)]]))
round.(density(steady); digits=3)   # ≈ [0.25]
```

The balance of loss rate `0.6` and gain rate `0.2` fixes the steady occupation at
`0.2 / (0.2 + 0.6) = 0.25`.

`CorrelationLindblad` is the number-conserving specialization of `MajoranaLindblad`,
which additionally handles pairing baths; see [BdG Pairing](bdg-pairing.md). The same
dynamics can be unravelled into [trajectories](trajectories-vs-lindblad.md).
````

- [ ] **Step 2: Snippet check**

Run:
```bash
julia --project=docs -e 'using GaussianFermions; s=steadystate(CorrelationLindblad(zeros(ComplexF64,1,1); loss_ops=[ComplexF64[sqrt(0.6)]], gain_ops=[ComplexF64[sqrt(0.2)]])); println(round.(density(s);digits=3))'
```
Expected: `[0.25]`.

- [ ] **Step 3: Commit**

```bash
git add docs/src/examples/deterministic-lindblad.md
git commit -m "docs: deepen deterministic-lindblad example with steady states and channels"
```

### Task 9: Deepen `trajectories-vs-lindblad.md`

**Files:**
- Modify: `docs/src/examples/trajectories-vs-lindblad.md` (full rewrite)

- [ ] **Step 1: Replace the file** (4-way density convergence from `Dephase.jl`)

````markdown
# Trajectories vs Lindblad

The deterministic Lindblad equation and its stochastic unravelings describe the same
average dynamics of **linear** observables when the channel conventions match.
GaussianFermions.jl leaves the sampling loop to the caller; this page shows the site
density `n(t)` agreeing across four engines on a hopping chain with occupation
monitoring `D[√γ nᵢ]`.

## The MCWF step

```julia
using GaussianFermions, LinearAlgebra, Random

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
```

## Deterministic references

The Dirac (`CorrelationLindblad`) and Majorana (`MajoranaLindblad`) master equations
agree with each other to machine precision on number-conserving channels:

```@example tvl
using GaussianFermions, LinearAlgebra

L = 12; γ = 0.5; dt = 0.1; T = 4.0
chans = [dephasing(i, L; γ=γ) for i in 1:L]

dirac = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))
lindD = CorrelationLindblad(hopping(L; pbc=false), chans)

majo  = MajoranaState(CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left")))
lindM = MajoranaLindblad(hopping(L; pbc=false), chans)

for _ in 1:round(Int, T / dt)
    evolve!(dirac, lindD, dt)
    evolve!(majo,  lindM, dt)
end
round(norm(density(dirac) - density(majo)); digits=12)
```

## Trajectory averages converge to the master equation

A Monte-Carlo wave-function average of the same density approaches the deterministic
result as `1/√ntraj` (stochastic — fixed seed for reproducibility):

```julia
function mcwf_density(; L=12, γ=0.5, dt=0.1, T=4.0, ntraj=400, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=false), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    acc = zeros(L)
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="left")
        for _ in 1:round(Int, T / dt); mcwf_step!(s, U, chans, dt; rng); end
        acc .+= density(s)
    end
    acc ./ ntraj
end

# ‖mcwf_density() − density(dirac)‖ ≈ 0.05  (≈ 1/√ntraj sampling error)
```

`example/Dephase.jl` runs all four engines — MCWF, quantum-state diffusion (QSD),
Dirac Lindblad, and Majorana Lindblad — and prints the pairwise differences. The
[Monitoring Protocols](monitoring-protocols.md) example contrasts the same
unravelings on a **nonlinear** observable, where they no longer coincide.
````

- [ ] **Step 2: Snippet check** (the deterministic agreement)

Run:
```bash
julia --project=docs -e 'using GaussianFermions, LinearAlgebra; L=12;γ=0.5;dt=0.1;T=4.0; ch=[dephasing(i,L;γ=γ) for i in 1:L]; d=CorrelationState(SlaterState(L=L,N=L÷2,config="left")); lD=CorrelationLindblad(hopping(L;pbc=false),ch); m=MajoranaState(CorrelationState(SlaterState(L=L,N=L÷2,config="left"))); lM=MajoranaLindblad(hopping(L;pbc=false),ch); for _ in 1:round(Int,T/dt); evolve!(d,lD,dt); evolve!(m,lM,dt); end; println(round(norm(density(d)-density(m));digits=12))'
```
Expected: a number `< 1e-9` (machine-precision agreement).

- [ ] **Step 3: Commit**

```bash
git add docs/src/examples/trajectories-vs-lindblad.md
git commit -m "docs: deepen trajectories-vs-lindblad with 4-engine density comparison"
```

### Task 10: Deepen `bdg-pairing.md`

**Files:**
- Modify: `docs/src/examples/bdg-pairing.md` (full rewrite)

- [ ] **Step 1: Replace the file**

````markdown
# BdG Pairing

Use `MajoranaState` and `BdGHamiltonian` whenever the Hamiltonian or a bath can
generate anomalous correlations `Fᵢⱼ = ⟨cᵢcⱼ⟩`. `A` is the Hermitian hopping block,
`B` the antisymmetric pairing block.

## Unitary pairing dynamics

Starting from the empty state, a pairing Hamiltonian builds up anomalous
correlations:

```@example bdg_pairing
using GaussianFermions, LinearAlgebra

A = zeros(ComplexF64, 4, 4)
B = zeros(ComplexF64, 4, 4)
B[1, 2] = 0.35; B[2, 1] = -0.35
B[3, 4] = -0.2; B[4, 3] = 0.2

H = BdGHamiltonian(A, B)
s = MajoranaState([0, 0, 0, 0])

evolve!(s, H, 0.75)
(density        = round.(density(s); digits=3),
 anomalous_norm = round(norm(anomalous_correlation(s)); digits=4))
```

## Ground and thermal states

Ground and thermal states come from Nambu/Bogoliubov diagonalization:

```@example bdg_pairing
eps = quasiparticle_energies(H)
gs  = groundstate(H)
ρβ  = thermalstate(H; β=2.0)

(min_quasiparticle_energy = round(minimum(eps); digits=4),
 ground_particle_number   = round(particle_number(gs); digits=4),
 thermal_particle_number  = round(particle_number(ρβ); digits=4))
```

## A pairing bath

Unlike the number-conserving solver, `MajoranaLindblad` accepts pairing
(number-non-conserving) baths through Majorana-linear jump vectors, which generate
nonzero anomalous correlations even from a number-conserving initial state:

```@example bdg_pairing
H4   = hopping(4; pbc=true)
lind = MajoranaLindblad(H4; loss_ops=[sqrt(0.3) * ComplexF64[1, 0, 0, 0]],
                        dephasing_ops=[(ComplexF64[1, 1, 0, 0] / sqrt(2), 0.5)])

maj = MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2")))
evolve!(maj, lind, 1.0)
(density        = round.(density(maj); digits=3),
 anomalous_norm = round(norm(anomalous_correlation(maj)); digits=4))
```

For a topological p-wave Hamiltonian and its Majorana edge modes, see the
[Kitaev Chain](kitaev-chain.md) example.
````

- [ ] **Step 2: Snippet check**

Run:
```bash
julia --project=docs -e 'using GaussianFermions, LinearAlgebra; A=zeros(ComplexF64,4,4); B=zeros(ComplexF64,4,4); B[1,2]=0.35;B[2,1]=-0.35;B[3,4]=-0.2;B[4,3]=0.2; H=BdGHamiltonian(A,B); s=MajoranaState([0,0,0,0]); evolve!(s,H,0.75); println(round(norm(anomalous_correlation(s));digits=4))'
```
Expected: a positive number (≈ 0.4–0.6) — pairing built up.

- [ ] **Step 3: Commit**

```bash
git add docs/src/examples/bdg-pairing.md
git commit -m "docs: deepen bdg-pairing example with ground/thermal states and pairing bath"
```

### Task 11: Deepen `monitored-mutual-information.md` (add tripartite information)

**Files:**
- Modify: `docs/src/examples/monitored-mutual-information.md` (full rewrite)
- Modify: `docs/make.jl` (retitle this nav entry — done in Task 14's nav rewrite)

- [ ] **Step 1: Replace the file**

````markdown
# Mutual & Tripartite Information

Measurement-induced calculations average a nonlinear diagnostic over a monitored
ensemble or a late-time window. The mutual information `I(A:B)` and the tripartite
information `I₃(A:B:C)` of separated regions are the standard order parameters: as the
monitoring rate `γ` grows, correlations between distant regions are suppressed and
`I₃` collapses toward zero.

## Nodal occupation monitoring

```julia
using GaussianFermions, Random

# Nodal monitoring modes dᵢ⁺ = (cᵢ₋₁⁺ + cᵢ⁺ + cᵢ₊₁⁺)/√3, one per site (PBC).
function nodal_monitors(L; γ)
    V = ComplexF64[1, 1, 1] / sqrt(3)
    [OccupationMonitor(QuasiMode([mod(i-2, L)+1, i, mod(i, L)+1], V, L); γ=γ)
     for i in 1:L]
end

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
```

## Late-time mutual and tripartite information

`A`, `B`, `C` are three consecutive quarter-rings; the diagnostics are averaged over
the last quarter of each trajectory and over the ensemble (stochastic — fixed seed):

```julia
function info_vs_gamma(; L=16, γ=1.0, dt=0.1, tspan=10.0, ntraj=24, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=true), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    A = collect(1:(L ÷ 4)); B = collect((L ÷ 4 + 1):(L ÷ 2))
    C = collect((L ÷ 2 + 1):(3L ÷ 4))
    nsteps = round(Int, tspan / dt); tail = nsteps ÷ 4
    ΣI = 0.0; ΣI3 = 0.0; nsamp = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
            if k > nsteps - tail
                ΣI  += mutual_information(s, A, B)
                ΣI3 += tripartite_information(s, A, B, C)
                nsamp += 1
            end
        end
    end
    (mutual=ΣI / nsamp, tripartite=ΣI3 / nsamp)
end

for γ in (0.2, 1.0, 3.0)
    r = info_vs_gamma(; γ=γ)
    println("γ = $γ  ⟨I(A:B)⟩ ≈ $(round(r.mutual; digits=3))",
            "   ⟨I₃⟩ ≈ $(round(r.tripartite; digits=3))")
end
```

Representative output (small ensemble; real finite-size scaling needs larger `L` and
`ntraj`):

```text
γ = 0.2  ⟨I(A:B)⟩ ≈ 1.270   ⟨I₃⟩ ≈ -0.213
γ = 1.0  ⟨I(A:B)⟩ ≈ 0.952   ⟨I₃⟩ ≈ -0.024
γ = 3.0  ⟨I(A:B)⟩ ≈ 0.365   ⟨I₃⟩ ≈  0.001
```

Both `I(A:B)` and `|I₃|` shrink as monitoring strengthens; `I₃ → 0` in the
strongly-monitored (area-law) regime. The full sweep is in
[`example/MI.jl`](https://github.com/jayren3996/GaussianFermions.jl/blob/main/example/MI.jl).
See [Measurement-Induced Transition](measurement-induced-transition.md) for the
entanglement-entropy view of the same physics.
````

- [ ] **Step 2: Snippet check** (the diagnostics run and trend down)

Run:
```bash
julia --project=docs -e 'using GaussianFermions, Random; L=16; U=propagator(hopping(L;pbc=true),0.1); ch=[dephasing(i,L;γ=1.0) for i in 1:L]; rng=Xoshiro(1); s=SlaterState(L=L,N=L÷2,config="Z2"); for _ in 1:50; evolve!(s,U); for c in ch; p,w=jump_rate(c,s,0.1); rand(rng)<p ? (apply_click!(c,s,w); normalize!(s)) : apply_noclick!(c,s,0.1); end; end; println(round(mutual_information(s,1:4,5:8);digits=3), " ", round(tripartite_information(s,1:4,5:8,9:12);digits=3))'
```
Expected: two finite numbers print without error.

- [ ] **Step 3: Commit**

```bash
git add docs/src/examples/monitored-mutual-information.md
git commit -m "docs: add tripartite information to the monitored MI example"
```

### Task 12: New example — Kitaev Chain

**Files:**
- Create: `docs/src/examples/kitaev-chain.md`

- [ ] **Step 1: Write the page**

````markdown
# Kitaev Chain & Bogoliubov Spectrum

The Kitaev p-wave chain is the canonical topological BdG model. With `t = Δ` and open
boundaries, the bulk is gapped while two unpaired Majorana modes sit at the ends — an
exact zero-energy mode in the topological phase (`|μ| < 2t`) that disappears in the
trivial phase (`|μ| > 2t`).

In the `BdGHamiltonian(A, B)` convention, `A` carries the hopping `−t` and on-site
`−μ`, and `B` carries the p-wave pairing `Δ` (antisymmetric):

```@example kitaev
using GaussianFermions, LinearAlgebra

function kitaev_blocks(L; t=1.0, Δ=1.0, μ=0.0)
    A = zeros(ComplexF64, L, L)
    B = zeros(ComplexF64, L, L)
    for j in 1:L
        A[j, j] = -μ
    end
    for j in 1:L-1
        A[j, j+1] = -t; A[j+1, j] = -t      # hopping (Hermitian)
        B[j, j+1] =  Δ; B[j+1, j] = -Δ      # p-wave pairing (antisymmetric)
    end
    A, B
end
nothing # hide
```

## Topological vs trivial spectrum

```@example kitaev
Atop, Btop = kitaev_blocks(12; t=1.0, Δ=1.0, μ=0.0)   # topological
Atriv, Btriv = kitaev_blocks(12; t=1.0, Δ=1.0, μ=3.0) # trivial

εtop  = quasiparticle_energies(BdGHamiltonian(Atop, Btop))
εtriv = quasiparticle_energies(BdGHamiltonian(Atriv, Btriv))

(topological_lowest = round.(εtop[1:3]; digits=4),
 trivial_lowest     = round.(εtriv[1:3]; digits=4))
```

The topological spectrum has an exact zero mode (the Majorana edge pair) below a bulk
gap of `2t`; the trivial spectrum is fully gapped.

## Ground state and pairing correlations

!!! note "Zero-mode degeneracy"
    Exactly at `μ = 0` the ground state is parity-degenerate, so the Nambu occupation
    is ill-defined and `groundstate` raises an error. Evaluate it at a slightly
    detuned point (still topological), which selects a definite-parity ground state.

```@example kitaev
A, B = kitaev_blocks(12; t=1.0, Δ=1.0, μ=0.5)   # detuned, still topological
H = BdGHamiltonian(A, B)
gs = groundstate(H)

(particle_number = round(particle_number(gs); digits=3),
 anomalous_norm  = round(norm(anomalous_correlation(gs)); digits=3),
 edge_density    = round.(density(gs)[[1, 6, 12]]; digits=3))
```

The end sites carry slightly enhanced density relative to the bulk — a footprint of
the localized edge modes. A finite-temperature state interpolates toward the
maximally mixed state:

```@example kitaev
ρβ = thermalstate(H; β=5.0)
round(particle_number(ρβ); digits=3)
```

See [BdG Pairing](bdg-pairing.md) for general pairing dynamics and baths.
````

- [ ] **Step 2: Snippet check** (verified output: zero mode + groundstate at μ=0.5)

Run:
```bash
julia --project=docs -e '
using GaussianFermions, LinearAlgebra
function kb(L; t=1.0, Δ=1.0, μ=0.0)
  A=zeros(ComplexF64,L,L); B=zeros(ComplexF64,L,L)
  for j in 1:L; A[j,j]=-μ; end
  for j in 1:L-1; A[j,j+1]=-t;A[j+1,j]=-t;B[j,j+1]=Δ;B[j+1,j]=-Δ; end
  A,B
end
At,Bt=kb(12;μ=0.0); println("topo: ", round.(quasiparticle_energies(BdGHamiltonian(At,Bt))[1:3];digits=4))
A,B=kb(12;μ=0.5); H=BdGHamiltonian(A,B); gs=groundstate(H)
println("gs N=", round(particle_number(gs);digits=3), " anom=", round(norm(anomalous_correlation(gs));digits=3))'
```
Expected: `topo: [0.0, 2.0, 2.0]` and `gs N=6.883 anom=1.224`.

- [ ] **Step 3: Commit**

```bash
git add docs/src/examples/kitaev-chain.md
git commit -m "docs: add Kitaev chain / Bogoliubov spectrum example"
```

### Task 13: New example — Measurement-Induced Transition

**Files:**
- Create: `docs/src/examples/measurement-induced-transition.md`
- Create: `example/Transition.jl`

- [ ] **Step 1: Write the example script `example/Transition.jl`**

```julia
using GaussianFermions, Random

#--------------------------------------------------------------------------------
# Measurement-induced entanglement transition for monitored free fermions.
# Hopping chain + projective-style occupation monitoring at rate γ, unravelled as a
# quantum-jump (MCWF) trajectory. Entanglement is trajectory-nonlinear, so the
# steady-state entropy is a TRAJECTORY AVERAGE — it cannot be read from the
# deterministic Lindblad solver. As γ grows, ⟨S⟩ crosses over from a log-law (weak
# monitoring) to an area law (strong monitoring).
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

"Late-time, trajectory-averaged entanglement entropy of region 1:ℓ."
function steady_entropy(ℓ; L=16, γ=1.0, dt=0.1, tspan=12.0, ntraj=24, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=true), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    nsteps = round(Int, tspan / dt); tail = nsteps ÷ 4
    Σ = 0.0; nsamp = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
            k > nsteps - tail && (Σ += entanglement_entropy(s, 1:ℓ); nsamp += 1)
        end
    end
    Σ / nsamp
end

function main()
    println("half-chain entropy vs monitoring rate (L=16):")
    for γ in (0.1, 0.5, 1.0, 2.0)
        println("  γ = $γ\t⟨S(L/2)⟩ ≈ ", round(steady_entropy(8; γ=γ); digits=3))
    end
    println("subsystem-size scaling S(ℓ), L=24:")
    for (label, γ) in (("weak  ", 0.2), ("strong", 2.0))
        Ss = [round(steady_entropy(ℓ; L=24, γ=γ, ntraj=16); digits=3) for ℓ in 2:2:12]
        println("  $label (γ=$γ): ", Ss)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

- [ ] **Step 2: Write the page `docs/src/examples/measurement-induced-transition.md`**

````markdown
# Measurement-Induced Entanglement Transition

Continuously monitoring a free-fermion chain competes with its unitary spreading of
entanglement. At weak monitoring, entanglement grows with subsystem size (a log-law);
at strong monitoring, repeated measurements pin the state and entanglement saturates
(an area law). Because entanglement is **trajectory-nonlinear**, the steady-state
entropy is a *trajectory average* — it cannot be obtained from the deterministic
Lindblad solver.

This page uses the quantum-jump (MCWF) unraveling of site occupation monitoring at
rate `γ`; you own the loop:

```julia
using GaussianFermions, Random

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

function steady_entropy(ℓ; L=16, γ=1.0, dt=0.1, tspan=12.0, ntraj=24, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=true), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    nsteps = round(Int, tspan / dt); tail = nsteps ÷ 4
    Σ = 0.0; nsamp = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
            k > nsteps - tail && (Σ += entanglement_entropy(s, 1:ℓ); nsamp += 1)
        end
    end
    Σ / nsamp
end
```

## Half-chain entropy vs monitoring rate

```julia
for γ in (0.1, 0.5, 1.0, 2.0)
    println("γ = $γ\t⟨S(L/2)⟩ ≈ ", round(steady_entropy(8; γ=γ); digits=3))
end
```

```text
γ = 0.1   ⟨S(L/2)⟩ ≈ 2.968
γ = 0.5   ⟨S(L/2)⟩ ≈ 2.165
γ = 1.0   ⟨S(L/2)⟩ ≈ 1.472
γ = 2.0   ⟨S(L/2)⟩ ≈ 0.746
```

The steady-state entanglement falls monotonically as monitoring strengthens.

## Subsystem-size scaling

The cleaner signature is how the entropy grows with region size `ℓ`. At weak
monitoring `S(ℓ)` grows steadily with `ℓ`; at strong monitoring it grows far more
slowly, consistent with an approach to area-law saturation:

```julia
for (label, γ) in (("weak  ", 0.2), ("strong", 2.0))
    Ss = [round(steady_entropy(ℓ; L=24, γ=γ, ntraj=16); digits=3) for ℓ in 2:2:12]
    println("$label (γ=$γ): ", Ss)
end
```

```text
weak   (γ=0.2): [1.17, 2.091, 2.85, 3.43, 3.74, 3.802]
strong (γ=2.0): [0.542, 0.687, 0.736, 0.801, 0.864, 0.85]
```

!!! note "Illustrative sizes"
    These use small `L`, `ntraj`, and evolution times so they run quickly. A genuine
    finite-size-scaling study needs larger systems, more trajectories, and several
    `L` to locate the critical `γ`. The full run is in
    [`example/Transition.jl`](https://github.com/jayren3996/GaussianFermions.jl/blob/main/example/Transition.jl).

See [Mutual & Tripartite Information](monitored-mutual-information.md) for the
information-theoretic order parameter and [Monitoring Protocols](monitoring-protocols.md)
for how the unraveling choice affects the entanglement.
````

- [ ] **Step 3: Run the example script end to end**

Run:
```bash
julia --project=docs example/Transition.jl
```
Expected: prints the γ sweep (≈ 2.97 → 0.75) and the two scaling rows matching the values in the page (weak grows with ℓ, strong saturates). Exit code 0.

- [ ] **Step 4: Commit**

```bash
git add docs/src/examples/measurement-induced-transition.md example/Transition.jl
git commit -m "docs: add measurement-induced entanglement transition example"
```

### Task 14: New example — Monitoring Protocols + Examples nav

**Files:**
- Create: `docs/src/examples/monitoring-protocols.md`
- Create: `example/Protocols.jl`
- Modify: `docs/make.jl` (Examples nav block)

- [ ] **Step 1: Write `example/Protocols.jl`**

```julia
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
```

- [ ] **Step 2: Write `docs/src/examples/monitoring-protocols.md`**

````markdown
# Monitoring Protocols Compared

The same Lindbladian can be unravelled into trajectories in different ways.
GaussianFermions.jl provides three occupation-monitoring protocols:

- **projective** — [`measure!`](@ref) collapses `nᵢ` with Born statistics;
- **quantum-jump (MCWF)** — `OccupationMonitor` channels with click / no-click
  back-action;
- **quantum-state diffusion (QSD)** — [`weak_measure!`](@ref) applies the continuous
  weak filter `exp(α nᵢ)`.

A key fact: **linear** observables (like the site density) agree across all three
unravelings after averaging, and match the deterministic `CorrelationLindblad`. But
**entanglement is trajectory-nonlinear** — it depends on the unraveling, so the
protocols generally give *different* entropies, even though each individually shows
monitoring-induced suppression. The transition is robust as *phenomenology*, not
value-for-value.

## Convention matching

All three target the dephasing Lindbladian `D[√γ nᵢ]`. Projective measurement of
`nᵢ` at rate `γₚ` unravels `D[√(2γₚ) nᵢ]`, so to compare against `D[√γ nᵢ]` use
projective rate `γ/2` while the channel and QSD rates are `γ`.

## Linear observable: density agrees

```julia
using GaussianFermions, LinearAlgebra, Random

# (mcwf_step!, lindblad_density, mcwf_density, qsd_density, projective_density:
#  see example/Protocols.jl — reproduced there in full.)

ref = lindblad_density()
println("Lindblad   density[1:4] = ", round.(ref[1:4]; digits=3))
println("MCWF       ‖Δ‖ = ", round(norm(mcwf_density() - ref); digits=3))
println("QSD        ‖Δ‖ = ", round(norm(qsd_density() - ref); digits=3))
println("Projective ‖Δ‖ = ", round(norm(projective_density() - ref); digits=3))
```

```text
Lindblad   density[1:4] = [0.905, 0.896, 0.827, 0.746]
MCWF       ‖Δ‖ = 0.053
QSD        ‖Δ‖ = 0.040
Projective ‖Δ‖ = 0.038
```

All three unravelings reproduce the deterministic density; the residuals are pure
`1/√ntraj` sampling error.

## Nonlinear observable: entanglement differs

Accumulating `entanglement_entropy(s, 1:L÷2)` instead of density (on an `L=16` ring)
gives **different** trajectory-averaged entropies for the three protocols, even though
they unravel the same Lindbladian:

```text
γ=0.5  MCWF=2.165  QSD=1.812  projective=2.122
γ=1.0  MCWF=1.472  QSD=1.272  projective=1.404
γ=2.0  MCWF=0.746  QSD=0.677  projective=0.839
```

The three numbers never coincide, and their order is not even fixed — here QSD is
consistently the lowest, while MCWF and projective swap as `γ` grows. Each protocol
still shows monitoring-induced suppression (every column falls as `γ` increases).
This is expected: the unraveling encodes *which* measurement record is kept, and
entanglement is conditioned on that record, so — unlike the averaged density — it is
not unraveling-invariant.

The full runnable comparison is in
[`example/Protocols.jl`](https://github.com/jayren3996/GaussianFermions.jl/blob/main/example/Protocols.jl).
See [Trajectories vs Lindblad](trajectories-vs-lindblad.md) for the linear-observable
agreement in more detail, and [Measurement-Induced Transition](measurement-induced-transition.md)
for the entanglement transition itself.
````

- [ ] **Step 3: Run the example script**

Run:
```bash
julia --project=docs example/Protocols.jl
```
Expected: `Lindblad density[1:4] = [0.905, 0.896, 0.827, 0.746]`, the three `‖Δ‖` lines all `≤ ~0.06`, then three entanglement rows where the protocols give visibly different values (QSD lowest, e.g. `γ=1.0  MCWF=1.472  QSD=1.272  projective=1.404`). Exit code 0.

- [ ] **Step 4: Rewrite the Examples nav in `docs/make.jl`**

Replace the entire `"Examples" => [ ... ]` block with:
```julia
        "Examples" => [
            "Free-Fermion Chain" => "examples/free-fermion-chain.md",
            "Deterministic Lindblad" => "examples/deterministic-lindblad.md",
            "Trajectories vs Lindblad" => "examples/trajectories-vs-lindblad.md",
            "Monitoring Protocols" => "examples/monitoring-protocols.md",
            "Measurement-Induced Transition" => "examples/measurement-induced-transition.md",
            "Mutual & Tripartite Information" => "examples/monitored-mutual-information.md",
            "BdG Pairing" => "examples/bdg-pairing.md",
            "Kitaev Chain" => "examples/kitaev-chain.md",
        ],
```

- [ ] **Step 5: Build the full site**

Run:
```bash
julia --project=docs docs/make.jl
```
Expected: build succeeds; all eight example pages appear in the sidebar; no errored `@example` blocks, no broken `@ref`/links.

- [ ] **Step 6: Commit**

```bash
git add docs/src/examples/monitoring-protocols.md example/Protocols.jl docs/make.jl
git commit -m "docs: add monitoring-protocols example and finalize examples nav"
```

---

## Phase 4 — Finalize

### Task 15: README example list, final build, and acceptance review

**Files:**
- Modify: `README.md` (Examples section)

- [ ] **Step 1: Add the new scripts to the README Examples list**

In `README.md`, under `## 🗂 Examples`, add to the bulleted list of `example/` scripts:
```markdown
- [`Transition.jl`](example/Transition.jl) — measurement-induced entanglement transition (`⟨S⟩` vs monitoring rate).
- [`Protocols.jl`](example/Protocols.jl) — projective / MCWF / QSD unravelings of occupation monitoring.
```
And update the trailing "Worked, narrated versions" link to also mention the new pages if appropriate.

- [ ] **Step 2: Full clean build + warning scan**

Run:
```bash
julia --project=docs docs/make.jl 2>&1 | tee /tmp/docbuild.log | tail -5
grep -iE "warn|error|broken|missing docstring|not found" /tmp/docbuild.log || echo "no warnings"
```
Expected: build finishes; the grep prints `no warnings` (or only pre-existing, unrelated warnings you can identify). Fix any broken `@ref` or `@example` that surface (e.g. a missing docstring for a symbol referenced by `@docs` — add a concise docstring to its definition in `src/`).

- [ ] **Step 3: Acceptance review against the spec**

Confirm each acceptance criterion in
`docs/superpowers/specs/2026-06-03-documentation-depth-pass-design.md`:
- Home shows the logo hero; sidebar logo + favicon render in light and dark.
- Getting Started in nav with runnable blocks; the two stale pages gone; no inbound links to them (`grep -rn "majorana-bdg\|correlation-lindblad" docs/src` returns nothing).
- Deterministic snippets are `@example`; stochastic ones use fixed seeds.
- Manual pages have admonitions + cross-links.
- Eight example pages exist; MI page computes `tripartite_information`; the transition page shows `⟨S⟩` vs `γ` and `S(ℓ)` scaling; the protocols page states the linear-vs-nonlinear distinction.
- `docs/build` is **not** staged (it is gitignored — confirm `git status` shows no `docs/build` entries).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: list new transition/protocols scripts in README"
```

- [ ] **Step 5: Finish the branch**

Hand off to the `superpowers:finishing-a-development-branch` skill to choose how to
integrate `docs/documentation-depth-pass` (PR vs merge to `main`).

---

## Self-Review Notes

- **Spec coverage:** logo hero (T2), dark-variant decision (T2), Getting Started revive + nav (T3), stale-page deletion + link fix (T4), manual admonitions/`@example`/cross-links (T5–T6), deepen all five examples (T7–T11), tripartite info on MI page (T11), three new example pages incl. Kitaev + transition + protocols (T12–T14), example/*.jl scripts for the heavy ones (T13–T14), build verification (T1, phase builds, T15), README scripts list (T15). All spec sections map to a task.
- **No placeholders:** every code block is complete and was run against the live package; representative stochastic outputs are real measured values.
- **Type/name consistency:** `mcwf_step!`, `steady_entropy(ℓ; …)`, `kitaev_blocks(L; t, Δ, μ)`, `info_vs_gamma`, the four `*_density` functions, and the channel/Majorana `jump_rate` forms are used identically wherever they appear.
- **Known API constraints encoded:** no `parity` on `MajoranaState`; `groundstate` detuned off the exact zero mode; projective rate `γ/2` vs channel/QSD rate `γ`; `weak_measure!(s, qm, α)` (QuasiMode) for `SlaterState` vs `weak_measure!(s, i, α)` (site) for `MajoranaState`.
