"""Lookup tables between Qiskit operation names and MIMIQ gate classes.

The conversion layer in :mod:`mimiq_qiskit.converter` uses these tables
to translate gates without resorting to runtime ``isinstance`` chains.
Qiskit operations are keyed by their ``.name`` attribute (lowercase
canonical name, stable across import paths).
"""

from __future__ import annotations

from typing import Callable, Sequence

import mimiqcircuits as mc


# Qiskit name → factory(params) returning a MIMIQ operation.
# Params arrive as a list of floats (or symbolic floats — see converter).
QISKIT_TO_MIMIQ: dict[str, Callable[[Sequence[float]], mc.Operation]] = {
    # ── single-qubit, no params ────────────────────────────────────────
    "id": lambda p: mc.GateID(),
    "x": lambda p: mc.GateX(),
    "y": lambda p: mc.GateY(),
    "z": lambda p: mc.GateZ(),
    "h": lambda p: mc.GateH(),
    "s": lambda p: mc.GateS(),
    "sdg": lambda p: mc.GateSDG(),
    "t": lambda p: mc.GateT(),
    "tdg": lambda p: mc.GateTDG(),
    "sx": lambda p: mc.GateSX(),
    "sxdg": lambda p: mc.GateSXDG(),
    # ── single-qubit, parametric ───────────────────────────────────────
    "rx": lambda p: mc.GateRX(p[0]),
    "ry": lambda p: mc.GateRY(p[0]),
    "rz": lambda p: mc.GateRZ(p[0]),
    "p": lambda p: mc.GateP(p[0]),
    "u": lambda p: mc.GateU(p[0], p[1], p[2]),
    "u1": lambda p: mc.GateU1(p[0]),
    "u2": lambda p: mc.GateU2(p[0], p[1]),
    "u3": lambda p: mc.GateU3(p[0], p[1], p[2]),
    # ── two-qubit, no params ───────────────────────────────────────────
    "cx": lambda p: mc.GateCX(),
    "cnot": lambda p: mc.GateCX(),
    "cy": lambda p: mc.GateCY(),
    "cz": lambda p: mc.GateCZ(),
    "ch": lambda p: mc.GateCH(),
    "swap": lambda p: mc.GateSWAP(),
    "iswap": lambda p: mc.GateISWAP(),
    "cs": lambda p: mc.GateCS(),
    "csdg": lambda p: mc.GateCSDG(),
    "csx": lambda p: mc.GateCSX(),
    "ecr": lambda p: mc.GateECR(),
    "dcx": lambda p: mc.GateDCX(),
    # ── two-qubit, parametric ──────────────────────────────────────────
    "cp": lambda p: mc.GateCP(p[0]),
    "crx": lambda p: mc.GateCRX(p[0]),
    "cry": lambda p: mc.GateCRY(p[0]),
    "crz": lambda p: mc.GateCRZ(p[0]),
    "cu": lambda p: mc.GateCU(p[0], p[1], p[2], p[3]),
    "cu1": lambda p: mc.GateCP(p[0]),
    "rxx": lambda p: mc.GateRXX(p[0]),
    "ryy": lambda p: mc.GateRYY(p[0]),
    "rzz": lambda p: mc.GateRZZ(p[0]),
    "rzx": lambda p: mc.GateRZX(p[0]),
    # ── three-qubit ────────────────────────────────────────────────────
    "ccx": lambda p: mc.GateCCX(),
    "toffoli": lambda p: mc.GateCCX(),
    "cswap": lambda p: mc.GateCSWAP(),
    "fredkin": lambda p: mc.GateCSWAP(),
    "ccz": lambda p: mc.Control(2, mc.GateZ()),
    # ── four-qubit ─────────────────────────────────────────────────────
    "c3x": lambda p: mc.GateC3X(),
}


# MIMIQ Gate class → (qiskit gate class, num_params).
# Used by the reverse converter. Mapping to concrete Qiskit gate classes
# (rather than an opaque ``Gate(name=...)``) means the produced circuits
# carry a real definition and can be transpiled and simulated by Qiskit,
# not just inspected. ``GateU1/U2/U3`` map onto their phase/``U``
# equivalents to avoid Qiskit's deprecated ``U1/U2/U3`` classes.
from qiskit.circuit.library import (  # noqa: E402
    C3XGate,
    CCXGate,
    CHGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CSdgGate,
    CSGate,
    CSwapGate,
    CSXGate,
    CUGate,
    CXGate,
    CYGate,
    CZGate,
    DCXGate,
    ECRGate,
    HGate,
    IGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    UGate,
    XGate,
    YGate,
    ZGate,
    iSwapGate,
)

MIMIQ_TO_QISKIT: dict[type, tuple[Callable[..., object], int]] = {
    mc.GateID: (IGate, 0),
    mc.GateX: (XGate, 0),
    mc.GateY: (YGate, 0),
    mc.GateZ: (ZGate, 0),
    mc.GateH: (HGate, 0),
    mc.GateS: (SGate, 0),
    mc.GateSDG: (SdgGate, 0),
    mc.GateT: (TGate, 0),
    mc.GateTDG: (TdgGate, 0),
    mc.GateSX: (SXGate, 0),
    mc.GateSXDG: (SXdgGate, 0),
    mc.GateRX: (RXGate, 1),
    mc.GateRY: (RYGate, 1),
    mc.GateRZ: (RZGate, 1),
    mc.GateP: (PhaseGate, 1),
    mc.GateU: (UGate, 3),
    mc.GateU1: (PhaseGate, 1),
    mc.GateU2: (lambda phi, lam: UGate(3.141592653589793 / 2, phi, lam), 2),
    mc.GateU3: (UGate, 3),
    mc.GateCX: (CXGate, 0),
    mc.GateCY: (CYGate, 0),
    mc.GateCZ: (CZGate, 0),
    mc.GateCH: (CHGate, 0),
    mc.GateSWAP: (SwapGate, 0),
    mc.GateISWAP: (iSwapGate, 0),
    mc.GateCS: (CSGate, 0),
    mc.GateCSDG: (CSdgGate, 0),
    mc.GateCSX: (CSXGate, 0),
    mc.GateECR: (ECRGate, 0),
    mc.GateDCX: (DCXGate, 0),
    mc.GateCP: (CPhaseGate, 1),
    mc.GateCRX: (CRXGate, 1),
    mc.GateCRY: (CRYGate, 1),
    mc.GateCRZ: (CRZGate, 1),
    mc.GateCU: (CUGate, 4),
    mc.GateRXX: (RXXGate, 1),
    mc.GateRYY: (RYYGate, 1),
    mc.GateRZZ: (RZZGate, 1),
    mc.GateRZX: (RZXGate, 1),
    mc.GateCCX: (CCXGate, 0),
    mc.GateCSWAP: (CSwapGate, 0),
    mc.GateC3X: (C3XGate, 0),
}


def supported_qiskit_names() -> set[str]:
    """Names of Qiskit operations the converter recognises directly."""
    return set(QISKIT_TO_MIMIQ.keys()) | {
        "measure",
        "reset",
        "barrier",
        "unitary",
        "if_else",
    }
