"""Compute exact expectation values on MIMIQ with the Estimator primitive.

Run against the MIMIQ cloud::

    python examples/expectation_values.py

Unlike Qiskit's generic estimator, :class:`mimiq_qiskit.MimiqEstimatorV2`
evaluates the observable directly with MIMIQ's expectation-value engine,
so the result carries no shot noise. This example sweeps a rotation angle
and prints ``<Z>`` at each point — a cosine curve — using a single
parameterised circuit and one batched submission.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from mimiqlink import MimiqConnection

from mimiq_qiskit import MimiqBackend, MimiqEstimatorV2


def main() -> None:
    conn = MimiqConnection()
    conn.connect()

    estimator = MimiqEstimatorV2(MimiqBackend(conn))

    # A single-qubit rotation with a free parameter.
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    observable = SparsePauliOp(["Z"])

    # Sweep theta over [0, 2π). The bindings array gives the result its
    # shape; all points run in one MIMIQ job.
    angles = np.linspace(0.0, 2.0 * np.pi, 9).reshape(-1, 1)
    result = estimator.run([(qc, observable, angles)]).result()

    for angle, ev in zip(angles.ravel(), result[0].data.evs):
        print(f"theta={angle:5.2f}  <Z>={ev:+.4f}")


if __name__ == "__main__":
    main()
