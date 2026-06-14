<p align="center">
  <img src="logo.svg" alt="MIMIQ" width="380">
</p>

# mimiq-qiskit

[Qiskit](https://www.ibm.com/quantum/qiskit) bridge for
[MIMIQ](https://qperfect.io).

A drop-in Qiskit `BackendV2` that lets Qiskit users submit circuits to
the MIMIQ cloud, or to any local MIMIQ simulator, without leaving the
Qiskit ecosystem. Build your circuits with `QuantumCircuit`, hand them
to a `MimiqBackend`, get back a familiar `qiskit.result.Result`.

Highlights:

- **`MimiqBackend`**: a `BackendV2` covering the standard gate set,
  arbitrary `UnitaryGate`s, and single-gate mid-circuit conditionals.
  A batch of circuits is submitted as a single MIMIQ job.
- **`MimiqSamplerV2` / `MimiqEstimatorV2`**: native Qiskit V2
  primitives. The estimator computes expectation values exactly with
  MIMIQ's expectation-value engine, with no shot noise.
- MIMIQ-specific run options (`bonddim`, `entdim`, `timelimit`,
  `noisemodel`, and more) pass straight through `backend.run(...)`.

## Repository layout

```
mimiq-qiskit/
├── docs/        # Sphinx documentation
├── examples/    # Runnable examples
├── src/         # Python package
└── tests/       # Pytest suite
```

Qiskit is Python-only, so this repo has no Julia side; the package lives
at the repo root rather than under a nested `mimiq-qiskit-python/`
directory.

## Quick start

```python
from qiskit import QuantumCircuit
from mimiqlink import MimiqConnection
from mimiq_qiskit import MimiqBackend

# Authenticate against the MIMIQ cloud (opens a browser prompt).
conn = MimiqConnection()
conn.connect()

backend = MimiqBackend(conn)

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

job = backend.run(qc, shots=1000)
counts = job.result().get_counts()
print(counts)
```

Local MIMIQ simulators (anything implementing the
`mimiqcircuits.backends.Backend` interface) wrap the same way; pass the
backend instance instead of a connection:

```python
from mimiq_qiskit import MimiqBackend
from somewhere import MyLocalBackend

backend = MimiqBackend(MyLocalBackend())
```

### Primitives

Prefer the native primitives for sampling and expectation values. The
estimator is exact, with no shot noise:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from mimiq_qiskit import MimiqBackend, MimiqEstimatorV2

backend = MimiqBackend(conn)
estimator = MimiqEstimatorV2(backend)

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

result = estimator.run([(qc, SparsePauliOp(["ZZ", "XX"], [1.0, 0.5]))]).result()
print(result[0].data.evs)
```

See `examples/sampling.py` and `examples/expectation_values.py` for
runnable scripts.

## Installation

```bash
pip install mimiq-qiskit
```

## Development

Branch model: active development on `devel`, releases on `main`.

```bash
uv sync             # install dev environment
uv run pytest -sxv  # run the test suite
uv build            # build wheel + sdist
```

### Repositories and releases

Development happens on the private GitLab repository (`origin`); the public
GitHub mirror (`public`, `qperfect-io/mimiq-qiskit`) is kept in sync on the
`main` branch and the public version tags:

- `main` and `vX.Y.Z` tags are pushed to both `origin` and `public`.
- `vX.Y.Z-private` tags live only on `origin/devel` and never reach the
  public remote.

On the public repository, pushing a `vX.Y.Z` tag triggers the PyPI publish
and GitHub Pages documentation deploy. To develop against an unreleased
`mimiqcircuits`, add the sibling checkout locally without committing it:

```bash
uv add --editable ../mimiqcircuits-python
```
