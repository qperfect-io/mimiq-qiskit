"""Compute expectation values on MIMIQ with the Estimator primitive.

Run against the MIMIQ cloud::

    python examples/expectation_values.py

Unlike Qiskit's generic estimator, :class:`mimiq_qiskit.MimiqEstimatorV2`
reads each Pauli term straight off the simulator state instead of
estimating it from measurements, so a deterministic circuit carries no
shot noise at any term weight.

The example has two parts. The first sweeps a rotation angle on a
deterministic circuit, where one exact number per point is all there is to
report. The second adds a mid-circuit measurement, which leaves the
circuit ending in an *ensemble* rather than a state: there is no single
exact value any more, only an average, and the estimator has to be told
what to spend on estimating it.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from mimiqlink import MimiqConnection

from mimiq_qiskit import MimiqBackend, MimiqEstimatorV2


def deterministic_sweep(backend) -> None:
    """One exact value per parameter binding, in a single submission."""
    estimator = MimiqEstimatorV2(backend)

    # A single-qubit rotation with a free parameter.
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    observable = SparsePauliOp(["Z"])

    # Sweep theta over [0, 2pi). The bindings array gives the result its
    # shape; all points run in one MIMIQ job.
    angles = np.linspace(0.0, 2.0 * np.pi, 9).reshape(-1, 1)
    result = estimator.run([(qc, observable, angles)]).result()[0]

    print("exact, one value per angle:")
    for angle, ev in zip(angles.ravel(), result.data.evs):
        print(f"  theta={angle:5.2f}  <Z>={ev:+.4f}")
    print(f"  metadata: {result.metadata}\n")


def ensemble_average(backend) -> None:
    """The same observable on a circuit with no single exact value.

    ``h`` then a measurement collapses the qubit to ``|0>`` or ``|1>`` with
    equal probability, and the ``x`` flips it. So ``<Z>`` is -1 or +1 on
    any one trajectory, and 0 over the ensemble. The measurement is state
    preparation here, not readout: were it the last operation on the qubit
    the estimator would drop it, as Qiskit expects of a pub.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)

    observable = SparsePauliOp(["Z"])

    # Averaging trajectories: each one contributes an exact value, so only
    # the ensemble is sampled. `stds` is the standard error of the mean.
    averaged = MimiqEstimatorV2(backend, trajectories=2000)
    result = averaged.run([(qc, observable)]).result()[0]
    print(
        f"averaged:  <Z>={float(result.data.evs):+.4f} "
        f"+/- {float(result.data.stds):.4f}"
    )
    print(f"  metadata: {result.metadata}")

    # Sampling: rotate into the measurement basis, measure, average the
    # eigenvalues, exactly as hardware does. Same target, more variance,
    # because the observable is sampled on top of the ensemble.
    sampled = MimiqEstimatorV2(backend, shots=2000)
    result = sampled.run([(qc, observable)]).result()[0]
    print(
        f"sampled:   <Z>={float(result.data.evs):+.4f} "
        f"+/- {float(result.data.stds):.4f}"
    )

    # A single trajectory is unbiased but as noisy as the observable's own
    # range, so it takes an explicit method= to ask for one.
    single = MimiqEstimatorV2(backend, method="exact")
    result = single.run([(qc, observable)]).result()[0]
    print(
        f"one draw:  <Z>={float(result.data.evs):+.4f}  "
        "(a whole unit from the average)"
    )


def main() -> None:
    conn = MimiqConnection()
    conn.connect()
    backend = MimiqBackend(conn)

    deterministic_sweep(backend)
    ensemble_average(backend)


if __name__ == "__main__":
    main()
