"""Reading a local backend's Pauli terms off the evolved state.

A MIMIQ ``LocalBackend`` runs in this process and can be driven one step at a
time: prepare the circuit, evolve it once, then ask the state for each term.
That is strictly less work than the portable route, which pushes one
``ExpectationValue`` operation per term onto the circuit and reads the
z-register back — building, submitting and walking operations whose answers
the state could have given directly.

:func:`local_evaluator` turns such a backend into the ``evaluate`` callable
:func:`mimiq_qiskit.estimation.estimate_pub` takes. A backend qualifies when it
advertises ``expectation_state``: that token is what declares
:meth:`~mimiqcircuits.backends.Backend.expectation` to be implemented rather
than inherited as a raiser. A remote connection never qualifies, so the
portable route remains the only one every estimator must have.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from inspect import Parameter, signature

from mimiq_qiskit.backend import check_run_options
from mimiq_qiskit.estimation import TermValues
from mimiq_qiskit.observables import pauli_support


__all__ = ["supports_direct_terms", "local_evaluator"]


def supports_direct_terms(backend) -> bool:
    """Whether ``backend`` can serve :func:`local_evaluator`.

    ``backend`` is a MIMIQ backend object, or ``None`` when the estimator was
    handed a connection or a bare callable instead.
    """
    from mimiqcircuits.backends import LocalBackend

    if not isinstance(backend, LocalBackend):
        return False
    return "expectation_state" in backend.capabilities()


@lru_cache(maxsize=4096)
def _term_operator(label: str):
    """``(PauliString, qubits)`` for a Qiskit Pauli label.

    Cached because parsing a label is linear in the register width and a
    parameter sweep asks for the same labels on every row: a 1-RDM over 100
    qubits is 100 labels of 100 characters, re-parsed once per row otherwise.
    """
    import mimiqcircuits as mc

    support = pauli_support(label)
    return (
        mc.PauliString("".join(p for _, p in support)),
        tuple(qubit for qubit, _ in support),
    )


#: The circuit-preparation knobs ``_resolve_prep_passes`` takes. It requires
#: every one of them and holds no defaults of its own: those belong to
#: ``execute``, which is what the portable route calls.
_PREP_KNOBS = (
    "fuse",
    "fuse_threshold",
    "canonicaldecompose",
    "reorderqubits",
    "remove_swaps",
)


@lru_cache(maxsize=1)
def _base_execute_params():
    """``LocalBackend.execute``'s parameters, where the knob defaults live."""
    from mimiqcircuits.backends import LocalBackend

    return signature(LocalBackend.execute).parameters


def _prep_kwargs(backend, opts: dict) -> dict:
    """The preparation knobs to pass, with ``execute``'s defaults filled in.

    Read off ``backend.execute``'s signature rather than written out here, so
    the two routes cannot drift: a default changed on ``execute`` would
    otherwise leave this path preparing circuits differently from the
    submission it stands in for, with nothing to signal it.

    Args:
        backend: The local backend being driven.
        opts: The options the caller set, already filtered by
            :func:`~mimiq_qiskit.backend.check_run_options`.

    Returns:
        One entry per knob in :data:`_PREP_KNOBS`.

    Raises:
        TypeError: If neither ``execute`` nor ``LocalBackend.execute``
            defaults a knob, which leaves no value to prepare with.
    """
    params = signature(backend.execute).parameters
    base = _base_execute_params()
    kwargs = {}
    for knob in _PREP_KNOBS:
        if knob in opts:
            kwargs[knob] = opts[knob]
            continue
        if knob in params:
            default = params[knob].default
        else:
            # A backend may narrow `execute` to the knobs it implements.
            # One it does not name is one its own `execute` cannot set
            # either, so what stands in for the submission is what the base
            # class would have resolved with.
            default = base[knob].default if knob in base else Parameter.empty
        if default is Parameter.empty:
            raise TypeError(
                f"{type(backend).__name__}.execute declares no default for "
                f"{knob!r}, so the preparation pipeline cannot be resolved "
                "without one"
            )
        kwargs[knob] = default
    return kwargs


def _zero_based(permutation, num_qubits: int) -> list[int]:
    """``permutation`` as ``old index -> new index``, 0-based.

    ``apply_passes`` composes permutations as 1-based while the passes that
    produce them are 0-based, so the base is read off the values rather than
    assumed. A permutation whose base is ambiguous is the identity on one
    qubit, where the two readings agree anyway.
    """
    sites = list(permutation)
    if sorted(sites) == list(range(num_qubits)):
        return sites
    if sorted(sites) == list(range(1, num_qubits + 1)):
        return [site - 1 for site in sites]
    raise ValueError(
        f"a qubit relabelling returned {sites!r}, which is not a permutation "
        f"of {num_qubits} qubits"
    )


def _site_map(permutations, num_qubits: int) -> Callable[[int], int]:
    """Map a user-space qubit to the index it ends up on.

    A reorder relabels every qubit reference the circuit carries, which is
    why the portable route needs no mapping: its ``ExpectationValue``
    operations are in the circuit and are carried along. Terms evaluated
    afterwards are not, so they are mapped here instead.

    ``permutations`` is the relabellings in the order they were applied, any
    of them ``None`` for a step that changed nothing. Both steps exist
    because backends differ on where reordering lives: exaqt puts it in the
    pass pipeline, tensorweaver inside ``compile``, and either may use both.
    """
    steps = [
        _zero_based(permutation, num_qubits)
        for permutation in permutations
        if permutation is not None
    ]
    if not steps:
        return lambda qubit: qubit
    if len(steps) == 1:
        sites = steps[0]
        return lambda qubit: sites[qubit]

    def site(qubit: int) -> int:
        for sites in steps:
            qubit = sites[qubit]
        return qubit

    return site


def local_evaluator(
    backend, run_opts: dict, seed
) -> Callable[[Sequence], list]:
    """Build the ``evaluate`` callable for a local ``backend``.

    Args:
        backend: A :class:`~mimiqcircuits.backends.LocalBackend` that
            :func:`supports_direct_terms` accepts.
        run_opts: The estimator's MIMIQ run options. Only the
            circuit-preparation knobs are read here; the rest configure a
            submission this path never makes.
        seed: Seed forwarded to the backend. Expectation values consume no
            randomness, but a preparation pass may — the qubit-reorder search,
            for one — so seeding keeps the chosen layout reproducible.

    Returns:
        ``evaluate([(circuit, labels), ...]) -> list[TermValues]``, one entry
        per circuit and in the same order.
    """
    import mimiqcircuits as mc
    from mimiqcircuits.backends.fidelity import as_lower_bound
    from mimiqcircuits.backends.passes import PassContext, apply_passes

    # The backend and the options are both fixed here, so the pipeline they
    # mean is too. Resolved on the first batch rather than now, so that an
    # option the backend refuses raises where submitting it would — reading
    # `execute`'s signature is the expensive part, and it is done once.
    resolved: list = []

    def prep_pipeline():
        if not resolved:
            # A backend names the options it implements, and the two routes
            # have to agree on that list: an option `execute` would refuse
            # cannot quietly work here instead. `_resolve_prep_passes` is what
            # `execute` itself calls, so both read the knobs the same way.
            opts = check_run_options(backend, run_opts)
            resolved.append(
                backend._resolve_prep_passes(
                    None, **_prep_kwargs(backend, opts)
                )
            )
        return resolved[0]

    def evaluate(requests) -> list:
        passes = prep_pipeline()
        # Resolved once per batch, mirroring `execute` over a list of circuits.
        rngs = backend._resolve_rngs(seed, None)

        results = []
        for circuit, labels in requests:
            terms = [(label, *_term_operator(label)) for label in labels]
            # An observable may name a qubit no gate touches. The state is
            # sized from the circuit, so the circuit has to carry that qubit;
            # a barrier is the cheapest way to say so and evolves to nothing.
            width = max(
                [circuit.num_qubits()]
                + [q + 1 for _, _, qubits in terms for q in qubits]
            )
            if width > circuit.num_qubits():
                circuit.push(mc.Barrier(1), width - 1)

            ctx = PassContext(backend=backend, rng=rngs.pass_)
            prepared, pass_perm = apply_passes(passes, ctx, circuit)[:2]
            compiled = backend.compile(prepared)
            site = _site_map(
                (pass_perm, compiled.metadata.qubit_permutation),
                prepared.num_qubits(),
            )

            state = backend.build_state(
                prepared.num_qubits(),
                prepared.num_bits(),
                prepared.num_zvars(),
            )
            state, fidelity = backend.evolve(state, compiled, rng=rngs.noise)

            # Asked for in site order. It makes no difference to a state
            # vector, and lets a backend whose cost depends on where it last
            # looked — an MPS, with its canonical centre — walk the register
            # once instead of jumping about.
            placed = sorted(
                (tuple(site(q) for q in qubits), label, op)
                for label, op, qubits in terms
            )
            values = {
                label: backend.expectation(state, op, *sites).real
                for sites, label, op in placed
            }
            results.append(TermValues(values, (as_lower_bound(fidelity),)))
        return results

    return evaluate
