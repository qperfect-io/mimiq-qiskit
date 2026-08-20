"""Run Qiskit circuits on a local MIMIQ simulator, no credentials needed.

The other examples target the MIMIQ cloud, which means authenticating
first. This one wraps ``mimiq-exaqt``, a state-vector simulator published
on PyPI, so it runs straight from a checkout::

    pip install mimiq-exaqt
    python examples/local_simulation.py

Any class inheriting from ``mimiqcircuits.backends.Backend`` substitutes
for ``ExaqtQCS`` unchanged; only the constructor argument differs between
running locally and running on the cloud.
"""

import numpy as np
from exaqt import ExaqtQCS
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from mimiq_qiskit import MimiqBackend, MimiqEstimatorV2, MimiqSamplerV2


def bell_counts(backend: MimiqBackend) -> None:
    """Sample a Bell state through the plain ``backend.run`` path."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    counts = backend.run(qc, shots=2000, seed=1).result().get_counts()
    print(f"Bell counts: {counts}")


def ghz_sampler(backend: MimiqBackend) -> None:
    """Sample a 5-qubit GHZ state through the native sampler."""
    qc = QuantumCircuit(5)
    qc.h(0)
    for q in range(4):
        qc.cx(q, q + 1)
    qc.measure_all()  # creates the classical register named "meas"

    sampler = MimiqSamplerV2(backend, seed=2)
    counts = sampler.run([qc], shots=2000).result()[0].data.meas.get_counts()
    print(f"GHZ counts: {counts}")


def expectation_values(backend: MimiqBackend) -> None:
    """Evaluate an observable exactly, and check it against Qiskit."""
    qc = QuantumCircuit(2)
    qc.ry(0.7, 0)
    qc.rz(0.3, 1)
    qc.cx(0, 1)

    observable = SparsePauliOp(["ZZ", "XX", "IZ"], [1.0, 0.5, -0.25])

    estimator = MimiqEstimatorV2(backend)
    got = float(estimator.run([(qc, observable)]).result()[0].data.evs)
    want = float(np.real(Statevector(qc).expectation_value(observable)))

    print(f"<O> from MIMIQ:  {got:+.12f}")
    print(f"<O> from Qiskit: {want:+.12f}")
    print(f"agree: {abs(got - want) < 1e-9}")


def main() -> None:
    # 24 qubits is what the Target advertises to the transpiler; the
    # simulator itself is bounded by memory, not by this number.
    backend = MimiqBackend(ExaqtQCS(), name="exaqt", num_qubits=24)

    bell_counts(backend)
    ghz_sampler(backend)
    expectation_values(backend)


if __name__ == "__main__":
    main()
