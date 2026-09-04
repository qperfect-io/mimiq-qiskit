A complete example on a local simulator
=======================================

Everything in this package works the same against the MIMIQ cloud and
against a local MIMIQ simulator; only the argument you hand
:class:`~mimiq_qiskit.MimiqBackend` differs. This page runs the whole
surface — counts, both primitives, mid-circuit measurement, transpilation
— against `mimiq-exaqt <https://pypi.org/project/mimiq-exaqt/>`_, a
state-vector simulator published on PyPI, so every snippet below runs from
a fresh environment with no credentials and no network.

Working locally is also how you check a result you do not trust: a small
circuit has a Qiskit reference (``Statevector``, ``Operator``) to compare
against, and the last section does exactly that.

.. code-block:: bash

   pip install mimiq-qiskit mimiq-exaqt

Setting up the backend
----------------------

``ExaqtQCS`` is a ``mimiqcircuits`` ``LocalBackend``, which is one of the
shapes :class:`~mimiq_qiskit.MimiqBackend` accepts (see
:doc:`wrapping_local_backends`):

.. code-block:: python

   from exaqt import ExaqtQCS
   from mimiq_qiskit import MimiqBackend

   backend = MimiqBackend(ExaqtQCS(), name="exaqt", num_qubits=24)

``num_qubits`` is the width advertised to Qiskit's transpiler, not a limit
on ``run``: a simulator is bounded by memory, not by a coupling map. Set it
high enough to cover the circuits you intend to ``transpile`` against the
backend.

Counts from ``backend.run``
---------------------------

The plain path returns a ``qiskit.result.Result``, exactly as an Aer or IBM
backend would:

.. code-block:: python

   from qiskit import QuantumCircuit

   qc = QuantumCircuit(2, 2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure([0, 1], [0, 1])

   result = backend.run(qc, shots=1000, seed=42).result()
   print(result.get_counts())

.. code-block:: text

   {'11': 474, '00': 526}

``seed`` is forwarded to MIMIQ, so a seeded run repeats exactly. Note that
``backend.run`` returns as soon as the job is submitted — the work happens
on a background thread and ``result()`` is the blocking call. That matters
much more against the cloud than against a local simulator, but the shape
of the API is the same either way.

Whatever the simulator reports about the run comes back as result
metadata:

.. code-block:: python

   metadata = result.results[0].metadata
   print(metadata["simulator"], metadata["simulator_version"])
   print(sorted(metadata["timings"]))
   print(metadata["fidelities"])

.. code-block:: text

   Exaqt 0.2.0
   ['apply', 'compile', 'sample', 'total']
   [1.0]

A state-vector simulator is exact, hence the fidelity of 1.0. An
approximate backend such as MIMIQ's MPS engine reports a real estimate
here, which is the number to watch when you raise ``bonddim``.

Sampling with the native primitive
----------------------------------

:class:`~mimiq_qiskit.MimiqSamplerV2` reads MIMIQ's sampled bitstrings
into one :class:`~qiskit.primitives.containers.BitArray` per classical
register, so multi-register circuits come back already split:

.. code-block:: python

   from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
   from mimiq_qiskit import MimiqSamplerV2

   qr = QuantumRegister(3, "q")
   alice = ClassicalRegister(2, "alice")
   bob = ClassicalRegister(1, "bob")

   circuit = QuantumCircuit(qr, alice, bob)
   circuit.h(0)
   circuit.cx(0, 1)
   circuit.x(2)
   circuit.measure(qr[0], alice[0])
   circuit.measure(qr[1], alice[1])
   circuit.measure(qr[2], bob[0])

   data = MimiqSamplerV2(backend, seed=7).run([circuit], shots=1000).result()[0].data
   print("alice:", data.alice.get_counts())
   print("bob  :", data.bob.get_counts())

.. code-block:: text

   alice: {'00': 478, '11': 522}
   bob  : {'1': 1000}

Each register is addressed by name on the result's ``data``. Alongside
``get_counts()`` a ``BitArray`` also gives you the raw per-shot array, if
you would rather post-process the shots than a histogram.

Exact expectation values
------------------------

:class:`~mimiq_qiskit.MimiqEstimatorV2` does not sample. It reads each
Pauli term off the state itself, so the answer carries no shot noise and
the reported standard deviations are zero:

.. code-block:: python

   from qiskit.quantum_info import SparsePauliOp
   from mimiq_qiskit import MimiqEstimatorV2

   psi = QuantumCircuit(2)
   psi.ry(0.7, 0)
   psi.cx(0, 1)

   estimator = MimiqEstimatorV2(backend)
   pub_result = estimator.run([(psi, SparsePauliOp(["ZZ", "XX"], [1.0, 0.5]))]).result()[0]
   print(pub_result.data.evs, pub_result.data.stds)

.. code-block:: text

   1.3221088436188455 0.0

That "no shot noise" is not a cosmetic difference. It means a scan over a
parameter gives a smooth curve you can descend, instead of a noisy one you
have to average out. Here is a two-qubit Hamiltonian, a one-parameter
ansatz, and a coarse-to-fine scan that closes on the ground state:

.. code-block:: python

   import numpy as np
   from qiskit.circuit import Parameter

   hamiltonian = SparsePauliOp(
       ["II", "ZI", "IZ", "ZZ", "XX"],
       [-1.05, 0.40, -0.40, 0.18, 0.18],
   )

   theta = Parameter("theta")
   ansatz = QuantumCircuit(2)
   ansatz.x(0)
   ansatz.ry(theta, 1)
   ansatz.cx(1, 0)

   estimator = MimiqEstimatorV2(backend)

   def scan(grid):
       """One estimator job covering every angle in ``grid``."""
       evs = estimator.run(
           [(ansatz, hamiltonian, [[angle] for angle in grid])]
       ).result()[0].data.evs
       best = int(np.argmin(evs))
       return grid[best], float(evs[best])

   low, high = -np.pi, np.pi
   for step in range(4):
       grid = np.linspace(low, high, 21)
       angle, energy = scan(grid)
       print(f"pass {step + 1}: E = {energy:+.9f} at theta = {angle:+.6f}")
       width = (high - low) / 20
       low, high = angle - width, angle + width

   exact = min(np.linalg.eigvalsh(hamiltonian.to_matrix()).real)
   print(f"exact ground state: {exact:+.9f}")

.. code-block:: text

   pass 1: E = -2.046468272 at theta = -2.827433
   pass 2: E = -2.049999193 at theta = -2.921681
   pass 3: E = -2.049999193 at theta = -2.921681
   pass 4: E = -2.049999991 at theta = -2.920425
   exact ground state: -2.050000000

Note what each ``scan`` call costs: passing an array of parameter bindings
as the third element of the pub batches all 21 circuits into **one** MIMIQ
submission, not 21. Against the cloud that is the difference between one
round trip per pass and twenty-one.

Mid-circuit measurement
-----------------------

A Qiskit ``if_test`` block holding a single gate converts to a MIMIQ
``IfStatement``, so measurement feed-forward works:

.. code-block:: python

   qc = QuantumCircuit(2, 2)
   qc.h(0)
   qc.measure(0, 0)
   with qc.if_test((qc.clbits[0], 1)):
       qc.x(1)
   qc.measure(1, 1)

   print(backend.run(qc, shots=1000, seed=3).result().get_counts())

.. code-block:: text

   {'11': 525, '00': 475}

The second qubit tracks the first, which is what a working conditional
looks like: no ``01`` or ``10``. Bodies with an ``else`` branch, several
instructions, or nested classical operations raise
:class:`~mimiq_qiskit.converter.UnsupportedGateError`; split them into
separate conditionals.

Gates outside the advertised Target
-----------------------------------

The ``Target`` lists the gates the transpiler may target, but the
converter accepts considerably more than that — see :doc:`conversion`. So
you have a choice, and it is worth knowing which way is cheaper.

.. code-block:: python

   from qiskit.circuit.library import XXPlusYYGate
   from qiskit import transpile
   from mimiq_qiskit import qiskit_to_mimiq

   qc = QuantumCircuit(5, 5)
   qc.h(range(4))
   qc.append(XXPlusYYGate(0.6, 0.2), [0, 1])   # not on the Target
   qc.mcx([0, 1, 2, 3], 4)                     # not on the Target
   qc.measure(range(5), range(5))

   print("xx_plus_yy on Target:", "xx_plus_yy" in backend.target.operation_names)
   print("direct    :", len(list(qiskit_to_mimiq(qc))), "MIMIQ instructions")

   transpiled = transpile(qc, backend=backend, seed_transpiler=1)
   print("transpiled:", len(list(qiskit_to_mimiq(transpiled))), "MIMIQ instructions")

   counts_direct = backend.run(qc, shots=4000, seed=11).result().get_counts()
   counts_transpiled = backend.run(transpiled, shots=4000, seed=11).result().get_counts()
   print("same result:", counts_direct == counts_transpiled)

.. code-block:: text

   xx_plus_yy on Target: False
   direct    : 11 MIMIQ instructions
   transpiled: 55 MIMIQ instructions
   same result: True

Both give the same distribution, but transpiling first is five times
larger. Qiskit synthesises the four-control ``mcx`` into primitives,
whereas the converter hands MIMIQ a native ``Control`` and lets MIMIQ
decompose it — which it can do better, knowing its own simulator.

So run circuits directly unless you have a reason to transpile. The
``Target`` earns its place when you want Qiskit's optimisation passes, or
when a pass in your own pipeline insists on a concrete basis.

Checking a result against Qiskit
--------------------------------

At small widths Qiskit computes the same quantities independently, which
makes a local simulator the right place to confirm a circuit means what
you think:

.. code-block:: python

   from qiskit.quantum_info import Statevector

   probe = QuantumCircuit(3)
   probe.ry(0.7, 0)
   probe.rx(0.3, 1)
   probe.cx(0, 1)
   probe.cry(1.1, 1, 2)

   observable = SparsePauliOp(
       ["ZZI", "IXX", "ZII", "IIY"], [1.0, 0.5, -0.25, 0.4]
   )

   got = float(MimiqEstimatorV2(backend).run([(probe, observable)]).result()[0].data.evs)
   want = float(np.real(Statevector(probe).expectation_value(observable)))

   print(f"MIMIQ  {got:+.12f}")
   print(f"Qiskit {want:+.12f}")

.. code-block:: text

   MIMIQ  +0.847260207297
   Qiskit +0.847260207297

They agree to machine precision because both are exact: the estimator
does no sampling, so there is no statistical band to allow for. This is
the check the test suite runs over every gate in the conversion map.

That holds for a circuit ending in a single state, which is every circuit
on this page. A circuit with a mid-circuit measurement, a reset, or a
noise model ends in an ensemble instead, and its expectation value is an
average the estimator has to be given a budget for. See
:ref:`estimation-methods`.

Moving to the cloud
-------------------

One line changes:

.. code-block:: python

   from mimiqlink import MimiqConnection

   conn = MimiqConnection()
   conn.connect()                      # opens a browser prompt

   backend = MimiqBackend(conn)        # everything above works unchanged

Two things to know when you do:

**Run options are not portable.** The cloud accepts every MIMIQ knob;
a local simulator names only the ones it implements. ``bonddim`` and the
other MPS settings mean nothing to a state-vector simulator, so passing
one to ``ExaqtQCS`` raises a ``ValueError`` naming the backend and the
option rather than failing somewhere inside mimiqcircuits. Set them only
on backends that have them. See :doc:`quickstart` for the full list.

**Batching is worth more.** ``backend.run([qc1, qc2, ...])`` and a pub
with an array of parameter bindings both become a single MIMIQ
submission. Locally that saves a little overhead; against the cloud it
saves a network round trip per circuit.

``examples/local_simulation.py`` in the repository is a runnable script
covering the same ground.
