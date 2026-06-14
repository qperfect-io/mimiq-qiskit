# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-06-14

### Added
- `MimiqSamplerV2` and `MimiqEstimatorV2`, native Qiskit V2 primitives.
  The estimator evaluates observables exactly with MIMIQ's
  expectation-value engine (no shot noise; standard deviations reported
  as zero); the sampler reads MIMIQ's sampled bitstrings straight into
  per-register bit arrays. Both batch a pub's circuits into a single
  MIMIQ submission.
- Conversion of Qiskit `UnitaryGate` to `mimiqcircuits.GateCustom`, so
  arbitrary matrix gates pass through without manual decomposition.
- Conversion of single-gate `IfElseOp` conditionals (mid-circuit
  measurement feed-forward) to `mimiqcircuits.IfStatement`. Bodies with
  an `else` branch, multiple instructions, or nested classical ops raise
  `UnsupportedGateError`. Execution of conditionals depends on
  server-side support.
- MIMIQ-specific run options (`bonddim`, `entdim`, `mpscutoff`,
  `mpsmethod`, `mpotraversal`, `timelimit`, `noisemodel`, `label`) are
  accepted by `MimiqBackend.run` and forwarded to MIMIQ when set.

### Changed
- `MimiqBackend.run` now submits a batch of circuits as a single MIMIQ
  job (MIMIQ accepts a list of circuits and returns a list of results)
  rather than one job per circuit.
- `mimiq_to_qiskit` emits concrete Qiskit gate classes instead of opaque
  named `Gate` instances, so reverse-converted circuits carry a real
  definition and can be transpiled and simulated.
- `mimiqcircuits` is now resolved from PyPI (`>=0.24`); the dev-only local
  path override was removed so public builds and CI resolve the dependency
  the same way users do.

### Fixed
- Results now carry per-shot `memory` alongside `counts`, so Qiskit's
  `BackendSamplerV2` reconstructs its bit arrays correctly instead of
  returning all-zero samples.

### CI
- Added GitHub Actions workflows for the public mirror: test matrix, docs
  build with GitHub Pages deploy on tags, and PyPI publish on `vX.Y.Z`
  tags.
- Added a `docs` dependency group; GitLab and GitHub docs builds now share
  it instead of installing Sphinx ad hoc.
