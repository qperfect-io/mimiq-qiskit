"""Sample a Bell state on MIMIQ with the native Sampler primitive.

Run against the MIMIQ cloud::

    python examples/sampling.py

The example builds a two-qubit entangling circuit, samples it through
:class:`mimiq_qiskit.MimiqSamplerV2`, and prints the measurement counts,
which concentrate on ``00`` and ``11`` for a Bell state.
"""

from qiskit import QuantumCircuit

from mimiqlink import MimiqConnection

from mimiq_qiskit import MimiqBackend, MimiqSamplerV2


def main() -> None:
    # Authenticate against the MIMIQ cloud (opens a browser prompt).
    conn = MimiqConnection()
    conn.connect()

    # The native sampler reads MIMIQ's sampled bitstrings directly.
    sampler = MimiqSamplerV2(MimiqBackend(conn))

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()  # creates the classical register named "meas"

    # One pub, 1000 shots. result()[0] is the result for that pub.
    result = sampler.run([qc], shots=1000).result()

    # `data.meas` is a BitArray over the "meas" register.
    counts = result[0].data.meas.get_counts()
    print(counts)


if __name__ == "__main__":
    main()
