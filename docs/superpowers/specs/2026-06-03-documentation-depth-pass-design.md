# Documentation Depth Pass Design

## Context

GaussianFermions.jl already has a Documenter.jl site produced by the previous
overhaul (see `2026-06-03-documentation-overhaul-design.md`). That overhaul
established a four-path structure — orientation, manual, examples, API reference —
and the acceptance criteria for it are largely met.

This pass is a **depth + polish** layer on top of that structure, not a restructure.
Three concrete gaps remain:

1. **Branding.** The README has a polished logo hero; the docs home page is plain
   text. Logo assets (`logo.png`, `logo.svg`, `favicon.ico`) already live in
   `docs/src/assets/` and Documenter auto-uses `logo.png`/`favicon.ico` for the
   sidebar, but there is no home-page hero and no dark-mode logo variant.
2. **Stale pages.** `docs/src/getting-started.md`,
   `docs/src/manual/majorana-bdg.md`, and `docs/src/manual/correlation-lindblad.md`
   exist on disk but are not in the `make.jl` nav. The two manual pages duplicate
   content now folded into `manual/lindblad.md`, `manual/hamiltonians.md`, and
   `manual/states.md`. `getting-started.md` is a genuinely useful guided tutorial
   that was dropped from the nav.
3. **Thin examples.** The five example pages are ~30–45 line snippets that defer to
   `example/*.jl` for the full workflow. The package's headline use case —
   **monitored free-fermion dynamics and measurement-induced entanglement
   transitions** — is only gestured at, even though every diagnostic it needs
   (`entanglement_entropy`, `renyi_entropy`, `mutual_information`,
   `tripartite_information`, `purity`) is already exported.

The target reader is unchanged: a physics researcher who already understands
Gaussian fermions and wants this package's conventions, modeling choices, API
contracts, and copyable workflows.

## Goals

- Add a logo hero to the docs home page and a dark-mode logo variant; verify the
  sidebar logo and favicon render.
- Revive `getting-started.md` as a proper nav page (a narrated first session) and
  remove the two stale, duplicated manual pages.
- Make every deterministic code snippet a build-verified `@example`; keep
  stochastic snippets as fixed-seed blocks. Add Documenter admonitions and
  consistent cross-links across the manual.
- Deepen the five existing example pages into self-contained, narrated workflows
  drawn from the `example/*.jl` scripts.
- Add three new example pages, weighted toward monitored dynamics and the
  measurement-induced entanglement transition (the package's headline use case):
  a measurement-induced transition page, a monitoring-protocol comparison page,
  and a Kitaev-chain BdG showcase.
- Expand the Monitored Mutual Information page into a transition-diagnostics page
  that adds tripartite information as the scale-invariant order parameter.
- Keep the build clean in CI; add docstrings only where a reference/`@docs` block
  would otherwise break.

## Non-Goals

- No change to the documentation system (stay on Documenter.jl) or the four-path
  structure.
- No API or physics changes. If a doc example exposes a genuine bug, it is fixed
  separately and noted, not silently worked around.
- No full docstring rewrite. The previous overhaul covered the public API; this
  pass only fills gaps that break `@docs`/reference pages.
- No rendered plot images. Examples are output-only (numbers and small tables),
  per the chosen scope. No new plotting dependency in `docs/Project.toml`.
- No purification-transition example this pass (considered and deferred).
- No hidden trajectory ensemble runner. Examples keep the caller-owned loop.

## Branding and Logo

- **Home hero.** Add a centered hero to the top of `docs/src/index.md` using a
  ```` ```@raw html ```` block: the logo (`assets/logo.svg`, ~180px), the tagline,
  and the existing badges, mirroring the README hero. The home prose follows.
- **Sidebar + favicon.** Documenter auto-detects `assets/logo.png` and
  `assets/favicon.ico`. Verify both render in the built site; no `make.jl` change
  needed for them unless verification shows otherwise.
- **Dark-mode variant.** Inspect `logo.svg`. If its strokes/text are illegible on a
  dark background, add `docs/src/assets/logo-dark.svg` (Documenter uses
  `logo-dark.*` automatically for dark themes). If the existing logo already reads
  on dark, skip this.

## Site Structure

- **Revive `getting-started.md`.** Rewrite it with verified `@example` blocks, fix
  its links (which currently point to the now-deleted pages), and add a short
  trajectory teaser that forwards to the manual. Place it in the nav between Home
  and the Manual:

  ```
  "Home" => "index.md",
  "Getting Started" => "getting-started.md",
  "Manual" => [ ... ],
  ```

- **Delete stale pages.** After confirming coverage, remove
  `docs/src/manual/majorana-bdg.md` and `docs/src/manual/correlation-lindblad.md`.
  Coverage check before deletion:
  - Unitary BdG dynamics, `groundstate`, `thermalstate`, `quasiparticle_energies`
    → must be in `manual/hamiltonians.md`.
  - `CorrelationLindblad`, `MajoranaLindblad`, loss/gain/dephasing/pairing baths,
    `steadystate`, `lindblad_rhs`/`majorana_lindblad_rhs` → must be in
    `manual/lindblad.md`.
  - Majorana state representation and conversions → must be in `manual/states.md`.
  Fold any unique sentences from the stale pages into these targets before
  deleting. Grep the tree for inbound links to the deleted files and fix them.

## Manual Deepening and Consistency

Apply across all six manual pages (`conventions`, `states`, `hamiltonians`,
`lindblad`, `trajectories`, `observables`):

- **Verified examples.** Convert deterministic ```` ```julia ```` blocks to
  ```` ```@example <pagename> ```` so the build runs them. `trajectories.md` is
  currently entirely unverified plain blocks; convert its deterministic setup
  snippets and keep the stochastic ones as fixed-seed blocks.
- **Admonitions.** Use Documenter admonitions consistently:
  - `!!! note` for convention reminders (index base, correlator/covariance
    conventions).
  - `!!! warning` for mutation (`!` functions mutate) and rate-vs-probability
    gotchas (Majorana `jump_rate` returns a rate, channel `jump_rate` returns a
    per-step probability).
- **Cross-links.** Ensure each manual page links forward to the relevant example(s)
  and to the matching API reference section, and that the home page and Getting
  Started link into the manual consistently.
- **Observables page.** Make sure `manual/observables.md` documents the full
  diagnostic set used by the new examples: `entanglement_entropy`,
  `renyi_entropy`, `entanglement_spectrum`, `mutual_information`,
  `tripartite_information`, `purity`, `parity`, `number_variance` — and notes which
  are linear (obtainable from the deterministic Lindblad) versus trajectory-nonlinear
  (only meaningful as a trajectory average).

## Examples

Eight example pages total: deepen the existing five, add three new ones. All pages
are self-contained (the full code is on the page). Heavier stochastic runs are
mirrored into a runnable `example/*.jl` script where one does not already exist,
following the existing repo convention (`FreeFermion.jl`, `MI.jl`, `Dephase.jl`).

### Snippet execution policy

- **Deterministic** snippets (spectra, ground/thermal states, deterministic
  Lindblad density, steady states) → ```` ```@example ```` blocks, verified by the
  build.
- **Stochastic / ensemble** snippets (trajectory averages of entanglement, MI,
  tripartite info) → fixed-seed (`Xoshiro(seed)`) ```` ```julia ```` blocks with a
  representative result shown in prose and a pointer to the full `example/*.jl`
  script. Rationale: trajectory-averaged entropies at small `ntraj` are not
  physically converged and can drift across Julia/BLAS versions, so they should not
  gate the build via `@example` doctest comparison.
- Build-executed runs use small sizes (`L ≈ 12–24`, `ntraj ≈ 16–50`, short
  `tspan`); each such page states the larger parameters needed for real
  finite-size scaling.

### Existing pages (deepen)

1. **Free-Fermion Chain** (`examples/free-fermion-chain.md`). Keep the deterministic
   entropy/density intro as `@example`. Add the full `mcwf_step!` definition and a
   trajectory-averaged half-chain entropy with a `1/√ntraj` error bar, from
   `FreeFermion.jl`. This page introduces the caller-owned trajectory loop reused by
   the monitored examples.
2. **Deterministic Lindblad** (`examples/deterministic-lindblad.md`). Absorb the
   useful content from the deleted `correlation-lindblad.md`: building a
   `CorrelationLindblad`, time evolution, `steadystate` for a finite attractive
   system, and `lindblad_rhs` for a custom integrator.
3. **Trajectories vs Lindblad** (`examples/trajectories-vs-lindblad.md`). Use
   `Dephase.jl`: MCWF, QSD, `CorrelationLindblad`, and `MajoranaLindblad` density
   evolutions agree; the deterministic pair agrees to machine precision and the
   trajectory averages approach it as `1/√ntraj`. Emphasize: this is a **linear**
   observable, which is why all unravelings coincide on average.
4. **BdG Pairing** (`examples/bdg-pairing.md`). Build a `BdGHamiltonian`, evolve a
   `MajoranaState`, read `normal_correlation`/`anomalous_correlation`, and obtain
   `groundstate`, `thermalstate`, `quasiparticle_energies`. Show a pairing
   (number-non-conserving) `MajoranaLindblad` bath generating nonzero anomalous
   correlations.
5. **Mutual & Tripartite Information** (`examples/monitored-mutual-information.md`,
   retitled). From `MI.jl`: nodal occupation monitoring of a hopping ring, late-time
   averaging. Add `tripartite_information` `I₃(A:B:C)` as the scale-invariant order
   parameter alongside `I(A:B)`, swept versus monitoring strength `γ`. State that
   `I₃` near zero / sign change is the cleaner numerical transition signature.

### New pages

6. **Kitaev Chain & Bogoliubov Spectrum** (`examples/kitaev-chain.md`). The one
   BdG/topology showcase. Build the p-wave Kitaev `BdGHamiltonian` (hopping +
   pairing + chemical potential) for both trivial and topological parameters;
   compute `quasiparticle_energies` (bulk gap; near-zero edge mode in the
   topological phase), `groundstate`, and `anomalous_correlation`; show a finite-`β`
   `thermalstate`. Deterministic → `@example` verified.
7. **Measurement-Induced Entanglement Transition** (`examples/measurement-induced-transition.md`).
   The monitored centerpiece. A monitored free-fermion chain (hopping + occupation
   monitoring at rate `γ`) evolved with the caller-owned MCWF loop from page 1.
   - Late-time, trajectory-averaged half-chain entropy `⟨S(L/2)⟩` versus `γ`,
     showing suppression as monitoring strengthens.
   - Subsystem-size scaling `⟨S(ℓ)⟩` vs `ℓ` at a weak and a strong `γ`,
     distinguishing log-law (weak) from area-law (strong).
   - Explicit note: entanglement is trajectory-nonlinear, so these come from the
     trajectory average and **cannot** be read off the deterministic Lindblad.
   - Mirror the full run into `example/Transition.jl`.
8. **Monitoring Protocols Compared** (`examples/monitoring-protocols.md`). The same
   monitored model through three unravelings — projective measurement (`measure!`),
   quantum-jump/MCWF (`jump_rate`/`apply_click!`/`apply_noclick!`), and QSD/weak
   measurement (`weak_measure!`).
   - Show the averaged **density** (a linear observable) agrees across all three and
     with the deterministic Lindblad.
   - Show the **entanglement entropy** dynamics differs between unravelings (it is
     trajectory-nonlinear and unraveling-dependent), while each protocol
     independently exhibits monitoring-induced entanglement suppression.
   - Frame "robustness" precisely: the *phenomenology* (a transition / crossover)
     is protocol-robust; exact entropy values are not unraveling-invariant. Avoid
     any claim that the three protocols give identical entanglement.
   - Mirror into `example/Protocols.jl` if the page run is heavy.

### `make.jl` Examples nav (final order)

```
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

## Build, Validation, and CI

- Instantiate the docs environment and build locally:

  ```bash
  julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
  julia --project=docs docs/make.jl
  ```

- The build must complete with no failed `@example` blocks, no broken internal
  links, and no missing-docstring errors from `@docs` blocks.
- Add or fix docstrings only where a reference/`@docs` block would otherwise error.
  A stricter `checkdocs` policy is out of scope for this pass; leave the existing
  `checkdocs = :none` unless tightening it is free.
- Do not commit `docs/build` artifacts.
- Do not overwrite unrelated uncommitted changes in
  `.github/workflows/documentation.yml` or `docs/make.jl`.

## Acceptance Criteria

- The docs home page shows a logo hero; the sidebar logo and favicon render; the
  sidebar logo is legible in both light and dark themes.
- `getting-started.md` is in the nav with verified examples and correct links; the
  two stale manual pages are deleted and nothing links to them.
- Every deterministic snippet across manual and example pages is an `@example`
  block; stochastic snippets use a fixed RNG seed.
- Manual pages use `note`/`warning` admonitions for conventions and mutation/rate
  gotchas, and cross-link to the matching examples and API reference.
- Eight example pages exist, including the three new ones; the Mutual & Tripartite
  Information page computes `tripartite_information`; the
  Measurement-Induced Transition page shows `⟨S⟩` vs `γ` and `S(ℓ)` scaling; the
  Monitoring Protocols page states the linear-vs-nonlinear distinction correctly.
- `julia --project=docs docs/make.jl` builds successfully with no errors.
- No `docs/build` artifacts are committed and no unrelated changes are overwritten.
