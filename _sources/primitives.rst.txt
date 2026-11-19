Primitives
==========

Qiskit's V2 primitives are the standard interface for sampling and
expectation-value workloads. ``mimiq-qiskit`` ships native
implementations that talk to MIMIQ directly, rather than relying on
Qiskit's generic ``BackendSamplerV2`` / ``BackendEstimatorV2`` wrappers.

The native primitives differ from the generic wrappers in two ways:

- :class:`~mimiq_qiskit.MimiqEstimatorV2` reads each Pauli term straight
  off the simulator state. The generic estimator can only estimate a term
  by sampling in a rotated basis; MIMIQ evaluates
  :math:`\langle \psi | P | \psi \rangle` directly, so a deterministic
  circuit carries no shot noise at any term weight.
- Both primitives submit every circuit in a pub as a single MIMIQ job
  instead of one round trip per circuit, and the estimator goes further:
  observables asked for at the same parameter binding are read from one
  evolution rather than one each.

Sampling
--------

.. code-block:: python

   from qiskit import QuantumCircuit
   from mimiqlink import MimiqConnection
   from mimiq_qiskit import MimiqBackend, MimiqSamplerV2

   conn = MimiqConnection(); conn.connect()
   sampler = MimiqSamplerV2(MimiqBackend(conn))

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure_all()

   result = sampler.run([qc], shots=1000).result()
   # Per-register bit arrays; ``measure_all`` writes the "meas" register.
   print(result[0].data.meas.get_counts())

Estimating expectation values
-----------------------------

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.quantum_info import SparsePauliOp
   from mimiq_qiskit import MimiqBackend, MimiqEstimatorV2

   estimator = MimiqEstimatorV2(MimiqBackend(conn))

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   observable = SparsePauliOp(["ZZ", "XX"], [1.0, 0.5])
   result = estimator.run([(qc, observable)]).result()
   print(result[0].data.evs)   # exact expectation value

Parameterised circuits broadcast the usual way: pass an array of bindings
as the third pub element, and the result's ``evs`` / bit arrays take the
broadcast shape.

An estimator pub carries state preparation, not readout, so a measurement
left at the end of the circuit is dropped rather than collapsing the state
the observable is about to be read from.

.. _estimation-methods:

Circuits that end in an ensemble
--------------------------------

The exactness above holds for a circuit that ends in one definite state.
A circuit with a mid-circuit measurement, a reset, a noise channel, or
qubit loss does not: MIMIQ re-evolves it once per shot, and each of those
*trajectories* has its own expectation value. The quantity you want,
:math:`\mathrm{Tr}(\rho O)`, is their average.

So the expectation value of such a circuit is a statistical estimate, and
the estimator needs a budget for it. Three methods are available,
selected by what you pass:

``trajectories=N``
   Average ``N`` trajectories. Each contributes an exact value, so only
   the ensemble is sampled, and ``stds`` reports the standard error of the
   mean. The most accurate use of a given budget.

``shots=N``
   Estimate from measurements in rotated bases, ``N`` shots per basis, the
   way hardware and Qiskit's ``BackendEstimatorV2`` do. Correct for any
   circuit, and the right choice for comparing against a shot-based
   reference, but for the same ``N`` it is noisier than averaging
   trajectories, because it samples the observable *and* the ensemble. Terms
   that commute qubit-wise share a basis, so an observable needing several
   bases costs ``N`` shots in each.

``precision=p``
   Size either of the above automatically, as ``ceil(1/p**2)``, following
   Qiskit's convention.

.. code-block:: python

   qc = QuantumCircuit(1, 1)
   qc.h(0)
   qc.measure(0, 0)     # state preparation, not readout: the x follows it
   qc.x(0)

   observable = SparsePauliOp(["Z"])

   # <Z> is -1 or +1 on any one trajectory, and 0 over the ensemble.
   averaged = MimiqEstimatorV2(backend, trajectories=4000)
   result = averaged.run([(qc, observable)]).result()[0]
   print(result.data.evs, "+/-", result.data.stds)

   # The same quantity, estimated the way a device would.
   sampled = MimiqEstimatorV2(backend, shots=4000)

Given no budget, a stochastic circuit raises rather than returning one
trajectory as though it were exact. ``method="exact"`` forces the
single-trajectory read anyway, with a warning: that value is an unbiased
draw, but its spread is the observable's own range, not a small error.

A noise model passed through ``run_options`` never appears in the
circuit, so the estimator cannot see it in the instructions. Any run
carrying ``noisemodel`` is therefore treated as stochastic.

Reading the metadata
--------------------

Each pub result says how its numbers were produced::

   {'target_precision': 0.0,
    'method': 'trajectories',   # or 'exact', 'shots'
    'exact': False,             # True only with no statistical error at all
    'stochastic': True,         # one evolution per shot was needed
    'trajectories': 4000,       # or 'shots': N
    'min_fidelity': 0.998}      # lowest simulator fidelity behind the pub

``min_fidelity`` is the simulator's own error, not a statistical one. On
the MPS backend it is the truncation fidelity, which averaging more
trajectories does not improve; raise ``bonddim`` for that.

Emulating shot noise cheaply
----------------------------

To see plausible shot noise on a deterministic circuit without paying a
shot budget, ``emulate_shot_noise=True`` adds Gaussian noise of width
``precision`` to the exact value, as Qiskit's ``StatevectorEstimator``
does. It costs one evolution. The flag is ignored when the value already
carries a real statistical error, since synthetic noise on top of that
would misreport the total.

.. code-block:: python

   estimator = MimiqEstimatorV2(backend, emulate_shot_noise=True, seed=7)
   result = estimator.run([(qc, observable)], precision=0.02).result()[0]
   print(result.data.evs, "+/-", result.data.stds)   # stds == 0.02
