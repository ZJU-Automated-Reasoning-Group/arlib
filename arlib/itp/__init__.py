"""
This module provides a proof assistant for the Z3 theorem prover.

It includes the following features:
- Proofs
- Axioms
- Definitions

(Note: here, "ITP" means "Interactive Theorem Proving", not "Interpolation")
"""

from . import smt
from . import kernel
from . import notation
from . import utils
from . import datatype
from . import rewrite
from . import tactics

Proof = kernel.Proof

prove = tactics.prove

axiom = kernel.axiom

define = kernel.define

QForAll = notation.QForAll

QExists = notation.QExists

cond = notation.cond

Inductive = kernel.Inductive

Struct = datatype.Struct

NewType = datatype.NewType

InductiveRel = datatype.InductiveRel

Enum = datatype.Enum

Calc = tactics.Calc

Lemma = tactics.Lemma

simp = rewrite.simp

search = utils.search

# TODO: Remove this
R = smt.RealSort()
Z = smt.IntSort()
RSeq = Z >> R
RFun = R >> R

__all__ = [
    "prove",
    "axiom",
    "define",
    "Proof",
    "QForAll",
    "QExists",
    "cond",
    "Struct",
    "NewType",
    "Inductive",
    "Calc",
    "Lemma",
    "R",
    "Z",
    "RSeq",
    "RFun",
    "simp",
    "search",
]
