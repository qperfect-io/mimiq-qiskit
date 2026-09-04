Quickstart
==========

Cloud execution
---------------

Authenticate against the MIMIQ cloud, wrap the connection in a
:class:`~mimiq_qiskit.MimiqBackend`, and run a familiar
``QuantumCircuit``:

.. code-block:: python

   from qiskit import QuantumCircuit
   from mimiqlink import MimiqConnection
   from mimiq_qiskit import MimiqBackend

   conn = MimiqConnection()
   conn.connect()

   backend = MimiqBackend(conn)

   qc = QuantumCircuit(2, 2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure([0, 1], [0, 1])

   job = backend.run(qc, shots=1000)
   counts = job.result().get_counts()

The job is executed in a background thread, so ``backend.run`` returns
immediately. Block for completion by calling ``job.result()``.

Provider helper
---------------

When you want a Qiskit-style provider entry point:

.. code-block:: python

   from mimiq_qiskit import MimiqProvider

   provider = MimiqProvider(conn)
   backend = provider.get_backend("mimiq")

Batching
--------

Pass a list of circuits to run them in one submission:

.. code-block:: python

   result = backend.run([qc_a, qc_b, qc_c], shots=2000).result()
   counts_a = result.get_counts(0)

MIMIQ run options
-----------------

``run`` forwards MIMIQ-specific options that have no Qiskit equivalent.
Circuit preparation happens before the state is evolved:

- ``fuse`` / ``fuse_threshold`` — merge runs of adjacent unitary gates
  into wider blocks, skipping circuits narrower than the threshold.
- ``canonicaldecompose`` — decompose the circuit to MIMIQ's canonical
  gate basis.
- ``reorderqubits``, ``remove_swaps`` — cloud-only layout passes.

The rest tune the simulator or the job itself: ``bonddim``, ``entdim``,
``mpscutoff``, ``mpsmethod``, ``mpotraversal`` for the MPS engine, plus
``timelimit``, ``noisemodel``, and ``label``.

.. code-block:: python

   job = backend.run(qc, shots=1000, fuse=True, bonddim=256)

Options left unset are not sent, so MIMIQ applies its own defaults. Note
that a local ``mimiqcircuits`` backend accepts only the preparation
knobs; passing a cloud-only option to one raises ``TypeError``.
