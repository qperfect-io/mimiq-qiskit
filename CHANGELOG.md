# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] — 2026-09-04

### Fixed
- `MimiqEstimatorV2` no longer reports a single trajectory as an exact
  expectation value. A circuit with a mid-circuit measurement, a reset, or a
  noise model is re-evolved per shot and each trajectory has its own value;
  the estimator submitted one shot and returned that one draw with
  `stds = 0` and `exact: True`. It now averages the trajectories, which is
  the unbiased estimate of `Tr(rho O)`, and reports the standard error of
  the mean.
- A stochastic circuit with no budget raises instead of returning one
  trajectory. Set `trajectories=N`, `shots=N`, or a positive `precision`;
  `method="exact"` still reads a single trajectory, with a warning.
- `MimiqEstimatorV2` drops measurements left at the end of a pub's circuit,
  as `TensorWeaverEstimator` already did. An estimator pub carries state
  preparation, not readout, so a trailing measurement would otherwise
  collapse the state the observable is read from.
- An identity-only observable returns its coefficient without running
  anything, instead of submitting a circuit whose result it cannot use.
- A non-Hermitian observable raises rather than having the imaginary part of
  its coefficients silently discarded.

### Added
- `MimiqEstimatorV2(shots=N)` estimates from measurements in rotated Pauli
  bases, the way hardware and Qiskit's `BackendEstimatorV2` do, for
  comparing against a shot-based reference. Terms are grouped into
  qubit-wise commuting sets, so one measurement circuit serves many terms.
- `MimiqEstimatorV2(trajectories=N)` averages N trajectories, and a positive
  `precision` sizes either budget as `ceil(1/precision**2)`, Qiskit's
  convention.
- `MimiqEstimatorV2(emulate_shot_noise=True)` adds Gaussian noise of width
  `precision` to an otherwise exact value, as Qiskit's
  `StatevectorEstimator` does. One evolution instead of a shot budget.
- `MimiqEstimatorV2(method=...)` forces `"exact"`, `"trajectories"`, or
  `"shots"` instead of letting `"auto"` choose from the circuit.
- Estimator result metadata reports `method`, `exact`, `stochastic`, the
  `trajectories` or `shots` spent, and `min_fidelity`, the lowest simulator
  fidelity behind the pub. On an MPS backend that is the truncation error,
  which averaging more trajectories does not reduce.
- `mimiq_qiskit.estimation` and `mimiq_qiskit.observables`, the estimation
  engine and the Pauli-observable helpers, so a provider-specific estimator
  can share one implementation of method resolution, trajectory averaging,
  and metadata rather than reimplementing them.

### Changed
- `MimiqEstimatorV2` reads observables through one `ExpectationValue`
  operation per Pauli term instead of `Circuit.push_expval`. Only a term's
  own qubits are named, so a weight-2 term on a wide register is a two-qubit
  operation rather than a full-width Pauli string padded with identities.
- Observables asked for at the same parameter binding share one submission
  and one evolution, and a term two of them share is evaluated once. Five
  observables on one circuit cost one MIMIQ job, not five.
- Estimator result metadata reports `target_precision` rather than
  `precision`, matching Qiskit's own estimators.
- `MimiqEstimatorV2.precision` is replaced by `default_precision`, settable
  in the constructor, again matching Qiskit's estimators.

## [0.2.0] — 2026-08-20

### Added
- `MimiqBackend.run` forwards MIMIQ's circuit-preparation knobs `fuse`,
  `fuse_threshold`, `canonicaldecompose`, `reorderqubits`, and
  `remove_swaps`, added in `mimiqcircuits` 0.26.3, alongside the
  simulator and job options it already accepted.
- `examples/local_simulation.py`, which runs on a local simulator rather
  than the cloud and so needs no credentials.
- `qiskit_to_mimiq` decomposes an unmapped gate through its own Qiskit
  `definition` instead of raising, so `initialize`, gates built with
  `QuantumCircuit.to_gate()`, open control states, and standard-library
  gates with no entry of their own now convert.
- A Qiskit `ControlledGate` with an all-ones control state becomes a
  native MIMIQ `Control` rather than being synthesised into primitives.
  A four-control `mcx` converts to one instruction where Qiskit's own
  definition would give 69.
- `mimiq_to_qiskit` falls back to a `UnitaryGate` carrying the
  operation's matrix for any unitary with no named counterpart, covering
  `Control`, `Inverse`, `Power`, `Parallel`, and MIMIQ-only gates such as
  `GateSY`, `GateHXY`, and `GateRNZ`. Non-unitary operations still raise
  rather than being dropped.
- `mimiq_to_qiskit` reconstructs a MIMIQ `IfStatement` as a Qiskit
  `if_test` block, so mid-circuit conditionals survive both directions.
- `GateR`, `GateXXplusYY`, `GateXXminusYY`, `GateISWAPDG`, and `GateCCP`
  map to and from their exact Qiskit counterparts, and Qiskit `delay`
  maps to MIMIQ `Delay` with its duration normalised to seconds.

### Changed
- Every dependency bound is now closed at the next version semantic
  versioning allows a breaking change in, so an incompatible release
  fails to resolve instead of failing at runtime. For a `0.y.z`
  dependency that boundary is the minor, because 0.x promises nothing
  across minors; from 1.0.0 on it is the major. `mimiqcircuits` is
  `>=0.26.7,<0.28` — 0.26.7 is where `GateCustom.matrix()` became
  callable again, and the range covers 0.27 because the suite runs
  against it too. `qiskit` becomes `>=2.0,<3`, and `numpy` `>=1.26,<3`,
  matching the range `mimiqcircuits` itself declares and raising a floor
  that could never resolve. Bounds widen only in a release whose suite
  has run against the new version, never on resolution alone.

  Overriding the constraint stays supported and is how the next bump gets
  validated: the GitLab pipeline resolves `mimiqcircuits` from the
  unreleased `devel` branch in one job, and `uv add --editable
  ../mimiqcircuits-python` does the same locally.

### Fixed
- A run option the target backend does not accept now raises a
  `ValueError` naming the backend and the option, instead of an opaque
  `TypeError: got an unexpected keyword argument` from inside
  mimiqcircuits. MIMIQ backends do not share one option set: the cloud
  takes every knob, a local simulator only the ones it implements.
- `MimiqSamplerV2` raises rather than reading past the end of a result
  when a backend returns fewer samples than the requested shots.
- Multi-qubit `UnitaryGate` conversion respected neither side's qubit
  ordering: Qiskit indexes a matrix little-endian (its first wire is the
  least significant bit) and MIMIQ big-endian, and the matrix was handed
  over unchanged on unchanged wires. A `UnitaryGate` holding a CX matrix
  became a CX with control and target swapped, silently. Both converters
  now reverse the wire order, which maps between the conventions without
  permuting the matrix. Single-qubit unitaries were never affected.
- `mimiq_to_qiskit` converts `GateCustom` again. `mimiqcircuits` 0.26.7
  restored `GateCustom.matrix` to a method, so reading it as an attribute
  raised `AttributeError`.

### Build
- `build-system.requires` is `uv_build>=0.11,<0.12`, which the running uv
  actually satisfies; the previous `<0.11.0` excluded it and warned on
  every build. CI installs a uv from the same range so the declared
  backend is the one that builds the wheel.
- The `dev` and `docs` dependency groups get the same treatment, with
  floors raised to the majors the suite runs on: `pytest>=9,<10`,
  `mimiq-exaqt>=0.2,<0.3`, `sphinx>=8,<9`,
  `sphinx-autodoc-typehints>=3,<4`. `furo` stays uncapped, being
  date-versioned with no breaking boundary to close at.

### CI
- Added a second GitLab test job that resolves `mimiqcircuits` from PyPI
  through the committed lock file. The existing job pins the unreleased
  GitLab `devel` branch, so nothing was exercising the dependency
  version published users install.
- The GitHub test workflow prints the resolved `mimiqcircuits`,
  `mimiq-exaqt`, and `qiskit` versions before running the suite.

### Docs
- Added "A complete example on a local simulator", a manual page working
  the whole API against `mimiq-exaqt`: counts and result metadata, both
  primitives, a coarse-to-fine parameter scan that closes on a
  Hamiltonian's ground state, mid-circuit measurement, running gates the
  advertised `Target` does not list, and a check against Qiskit's own
  `Statevector`. Every snippet runs without credentials, and its outputs
  are real transcripts.
- The quickstart documents the MIMIQ-specific run options accepted by
  `MimiqBackend.run`.
- Added a "What converts, and how" page describing the three-stage
  Qiskit-to-MIMIQ path, the reverse fallback, and the qubit-ordering
  conventions both directions bridge.

### Tests
- Added a simulation suite that runs converted circuits on
  `mimiq-exaqt`, a state-vector simulator on PyPI, and compares the
  resulting state to Qiskit's own `Statevector`. Every gate in the
  conversion map is checked this way, in both directions: a gate pointed
  at the wrong MIMIQ class, parameters passed in the wrong order, or a
  dropped qubit-ordering convention all produce well-formed but wrong
  results that a stubbed backend and a conversion round trip cannot see.
- Added tests for `GateCustomDiagonal` and for `fuse_circuit` output,
  the operations 0.27 introduces, so the declared `mimiqcircuits` range
  is backed by coverage rather than by the older tests happening to
  still pass. Both skip on a version that predates the feature, which is
  how a range spanning two minors stays honest.
- Added coverage for the job lifecycle (worker exceptions, timeouts,
  double submission, cancellation), the `QCSResults`-to-`Result`
  mapping (bit order, register layout, counts/memory agreement), and
  `MimiqProvider`, none of which had tests.
- The manual's worked example is executed as a test, so a snippet that
  stops working fails CI rather than sitting broken in the published
  documentation.

## [0.1.1] — 2026-06-14

### Docs
- Reviewed docstrings, comments, and documentation for clarity. Fixed the
  from-source install path and the `mimiqcircuits` version listed in the
  installation docs.

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
