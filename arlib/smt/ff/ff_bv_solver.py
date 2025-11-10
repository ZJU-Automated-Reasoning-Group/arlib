#!/usr/bin/env python3
"""
ffbv_solver.py  –  Finite-field solver via faithful BV encoding

Usage example
-------------
python ffbv_solver.py demo
python ffbv_solver.py regress  path/to/ff/benchmarks
"""
from __future__ import annotations
from typing import Dict, Optional, List
import sys, os, pathlib
import z3

# ---------  very small ad-hoc AST (replace with your real one)  ----------
class FieldExpr:
     pass

class FieldAdd(FieldExpr):
    def __init__(self, *args):
        self.args = list(args)

class FieldMul(FieldExpr):
    def __init__(self, *args):
        self.args = list(args)

class FieldEq (FieldExpr):
    def __init__(self, l, r):
        self.left, self.right = l, r

class FieldVar(FieldExpr):
    def __init__(self, name):
        self.name = name

class FieldConst(FieldExpr):
    def __init__(self, val):
        self.value = val

class ParsedFormula:
    def __init__(self, field_size:int,
                       variables:Dict[str,str],
                       assertions:List[FieldExpr]):
        self.field_size  = field_size
        self.variables   = variables    # name → sort id (unused here)
        self.assertions  = assertions

# -----------------------------------------------------------------------

class FFBVSolver:
    """
    Faithful translation of finite-field (prime field) constraints to QF_BV.
    All temporaries are evaluated in 2·k bits, with k = ⌈log₂ p⌉.
    """
    def __init__(self, theory:str="QF_BV"):
        self.solver         = z3.Solver()
        self.vars : Dict[str,z3.BitVecRef] = {}
        self.p              : Optional[int]        = None
        self.k              : Optional[int]        = None   # base width
        self.kw             : Optional[int]        = None   # 2*k width
        self.p_wide_bv      : Optional[z3.BitVecVal] = None

    # ----------------------  public API  -------------------------------
    def check(self, formula:ParsedFormula)->z3.CheckSatResult:
        self._setup_field(formula.field_size)
        self._declare_vars(formula.variables)

        for a in formula.assertions:
            self.solver.add(self._tr(a))

        return self.solver.check()

    def model(self)->Optional[z3.ModelRef]:
        if self.solver.reason_unknown():
            return None
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    # -----------------------  helpers  ---------------------------------
    def _setup_field(self, p:int)->None:
        if p <= 1:
            raise ValueError("Field size must be prime ≥ 2")
        self.p  = p
        self.k  = (p-1).bit_length()
        self.kw = self.k * 2
        self.p_wide_bv = z3.BitVecVal(p, self.kw)

    def _declare_vars(self, varmap:Dict[str,str])->None:
        for v in varmap:
            if v in self.vars: continue
            self.vars[v] = z3.BitVec(v, self.k)
            # 0 ≤ v < p
            self.solver.add(z3.ULT(self.vars[v], z3.BitVecVal(self.p, self.k)))

    # ----------  BV arithmetic with exact mod-p reduction  --------------
    def _to_wide(self, e:z3.BitVecRef)->z3.BitVecRef:
        return z3.ZeroExt(self.kw - e.size(), e)

    def _mod_p(self, wide:z3.BitVecRef)->z3.BitVecRef:
        """Reduce a kw-bit term modulo p and return a k-bit value."""
        tmp   = z3.URem(wide, self.p_wide_bv)     # kw bits
        return z3.Extract(self.k-1, 0, tmp)       # back to k bits

    # ----------------  recursive translation  --------------------------
    def _tr(self, e:FieldExpr)->z3.ExprRef:
        if isinstance(e, FieldAdd):
            wide = z3.BitVecVal(0, self.kw)
            for arg in e.args:
                wide = wide + self._to_wide(self._tr(arg))
            return self._mod_p(wide)

        if isinstance(e, FieldMul):
            wide = z3.BitVecVal(1, self.kw)
            for arg in e.args:
                wide = wide * self._to_wide(self._tr(arg))
            return self._mod_p(wide)

        if isinstance(e, FieldEq):
            return self._tr(e.left) == self._tr(e.right)

        if isinstance(e, FieldVar):
            return self.vars[e.name]

        if isinstance(e, FieldConst):
            if not (0 <= e.value < self.p):
                raise ValueError("Constant out of field range")
            return z3.BitVecVal(e.value, self.k)

        raise TypeError(f"Unexpected AST node {type(e).__name__}")

# -----------------------------------------------------------------------
# Dummy parser / demo ----------------------------------------------------
def tiny_demo() -> ParsedFormula:
    # m * x + 16 == is_zero     ∧     is_zero * x == 0
    p = 17
    x, m, z = "x", "m", "is_zero"
    vars  = {x:"ff", m:"ff", z:"ff"}
    f1 = FieldEq(
            FieldAdd(FieldMul(FieldVar(m), FieldVar(x)),
                     FieldConst(16),
                     FieldVar(z)),
            FieldConst(0))
    f2 = FieldEq(FieldMul(FieldVar(z), FieldVar(x)), FieldConst(0))
    return ParsedFormula(p, vars, [f1, f2])

def run_demo():
    solver = FFBVSolver()
    res = solver.check(tiny_demo())
    print("Result:", res)
    if res == z3.sat:
        print("Model:", solver.model())

# -----------------------------------------------------------------------
def regress(dir_path:str)->None:
    """Walk a directory containing .smt2 finite-field benchmarks that   │
       include one of the two meta-comments                            │
          ; EXPECT: sat        or   ; EXPECT: unsat                    """
    for fn in pathlib.Path(dir_path).rglob("*.smt2"):
        with fn.open() as f: txt = f.read()
        expect = "unknown"
        if "; EXPECT: sat"   in txt: expect="sat"
        if "; EXPECT: unsat" in txt: expect="unsat"

        # here you would call a *real* FF parser -----------------------
        formula = tiny_demo()        # placeholder
        # --------------------------------------------------------------

        s = FFBVSolver()
        got = s.check(formula)
        verdict = str(got)
        ok = verdict == expect or expect=="unknown"
        print(f"{fn.name:<30}  expect={expect:7}  got={verdict:7}  {'✓' if ok else '✗'}")

# -----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv)==2 and sys.argv[1]=="demo":
        run_demo()
    elif len(sys.argv)==3 and sys.argv[1]=="regress":
        regress(sys.argv[2])
    else:
        print("Usage:\n  ffbv_solver.py demo\n  ffbv_solver.py regress <dir>")
