What converts, and how
======================

:func:`~mimiq_qiskit.qiskit_to_mimiq` and
:func:`~mimiq_qiskit.mimiq_to_qiskit` are used by every execution path in
this package, and are exposed directly for callers who want a MIMIQ
circuit without running it.

Qiskit to MIMIQ
---------------

Conversion is tried in three stages, so the set of accepted circuits is
much wider than the list of named gates.

**1. Named gates.** Anything in
:data:`mimiq_qiskit.gate_map.QISKIT_TO_MIMIQ` maps to its MIMIQ
counterpart directly, along with ``measure``, ``reset``, ``barrier``,
``delay``, and ``unitary``. Call
:func:`~mimiq_qiskit.gate_map.supported_qiskit_names` for the current
list.

**2. Controlled gates.** A Qiskit ``ControlledGate`` whose base gate is
named, and whose control state is all-ones, becomes a MIMIQ ``Control``
carrying the control natively. This matters for width: Qiskit synthesises
a four-control ``mcx`` into 69 primitive gates, while MIMIQ keeps it as
one instruction and decomposes it its own way.

**3. The gate's own definition.** Anything left over is decomposed
through its Qiskit ``definition`` and each piece converted recursively.
That covers ``initialize``, gates built by ``QuantumCircuit.to_gate()``,
open control states, and any standard-library gate with no entry of its
own.

An operation that reaches stage three with no definition raises
:class:`~mimiq_qiskit.converter.UnsupportedGateError`.

Also handled:

- ``IfElseOp`` with a single-gate body and no ``else`` branch becomes a
  MIMIQ ``IfStatement``, which is how mid-circuit measurement
  feed-forward crosses over. Richer control flow (loops, ``switch``,
  multi-instruction bodies) raises.
- A ``GlobalPhaseGate``, and a circuit's ``global_phase``, are dropped.
  Neither affects a measurement distribution or an expectation value,
  which are the only quantities this bridge computes.
- Free :class:`~qiskit.circuit.Parameter` references are not resolved.
  Bind them with ``QuantumCircuit.assign_parameters`` first; the
  primitives do this for you.

MIMIQ to Qiskit
---------------

Named gates map back to concrete Qiskit gate classes, so the result is a
fully defined circuit Qiskit can transpile and simulate.

Any other **unitary** operation becomes a ``UnitaryGate`` holding its
matrix. That covers ``GateCustom``, the composite wrappers (``Control``,
``Inverse``, ``Power``, ``Parallel``), and MIMIQ gates Qiskit has no
equivalent for such as ``GateSY``, ``GateHXY``, or ``GateRNZ``. The
result is operator-faithful but not structure-faithful: a round trip
through both converters preserves the unitary, not the gate names.

Non-unitary MIMIQ operations with no Qiskit form — noise channels, above
all — raise :class:`~mimiq_qiskit.converter.UnsupportedGateError` rather
than being silently dropped.

Qubit ordering
--------------

Qiskit and MIMIQ disagree about which end of a multi-qubit matrix is
which. Qiskit indexes little-endian, so the first wire of a
``UnitaryGate`` is the *least* significant bit of its matrix; MIMIQ
indexes big-endian, so the first wire is the *most* significant.

Both converters bridge that by applying the matrix on reversed wires,
which is exactly equivalent to permuting its ``4**n`` entries and costs
nothing. Circuits are otherwise index-aligned: Qiskit qubit ``j`` is
MIMIQ qubit ``j``, and likewise for classical bits, using the flattened
index ``QuantumCircuit.find_bit(bit).index`` reports.

This is worth stating because getting it wrong is silent. A
``UnitaryGate`` holding a CX matrix converts into a perfectly valid
circuit with control and target exchanged, and a conversion round trip
cannot detect it, since both directions would be wrong the same way. The
test suite therefore checks converted circuits against Qiskit's
``Statevector`` on a real simulator, not against each other.
