#!/usr/bin/env python3
"""
ff_int_solver.py  –  Finite-field formulas via integer translation
Sound for every prime field GF(p).
"""
from __future__ import annotations
from typing import Dict, List, Optional
import math, sys, pathlib
import z3

# ---------- minimal AST (replace with your real classes) -------------
class FieldExpr:
     pass

class FieldAdd(FieldExpr):
    def __init__(self,*args):
        self.args = list(args)

class FieldMul(FieldExpr):
    def __init__(self,*args):
        self.args = list(args)

class FieldEq (FieldExpr):
    def __init__(self,l,r):
        self.left = l
        self.right = r

class FieldVar(FieldExpr):
    def __init__(self,name):
        self.name = name

class FieldConst(FieldExpr):
    def __init__(self,val):
        self.value = val

class ParsedFormula:
    def __init__(self, field_size:int,
                       variables:Dict[str,str],
                       assertions:List[FieldExpr]):
        self.field_size  = field_size
        self.variables   = variables
        self.assertions  = assertions
# ---------------------------------------------------------------------

def _is_prime(n:int)->bool:
    return n >= 2 and all(n % k for k in range(2, int(math.isqrt(n))+1))

class FFIntSolver:
    """
    Prime-field solver via a direct translation to non-linear integers.
    """
    def __init__(self):
        self.solver   = z3.SolverFor("QF_NIA")
        self.vars : Dict[str,z3.IntRef] = {}
        self.p    : Optional[int] = None

    # ---------------------------------------------------------------
    def check(self, formula:ParsedFormula)->z3.CheckSatResult:
        self._setup_field(formula.field_size)
        self._declare_vars(formula.variables)
        for a in formula.assertions:
            self.solver.add(self._tr(a))
        return self.solver.check()

    def model(self)->Optional[z3.ModelRef]:
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    # ---------------------------------------------------------------
    def _setup_field(self, p:int)->None:
        if not _is_prime(p):
            raise ValueError(f"Finite-field sort requires prime p, got {p}")
        self.p = p

    def _declare_vars(self, varmap:Dict[str,str])->None:
        for v in varmap:
            if v in self.vars: continue
            iv = z3.Int(v)
            self.vars[v] = iv
            self.solver.add(z3.And(iv >= 0, iv < self.p))

    # ---------------- translation helpers --------------------------
    def _mod(self, term:z3.IntRef)->z3.IntRef:
        return term % self.p             # ← portable remainder

    def _tr(self, e:FieldExpr)->z3.ExprRef:
        if isinstance(e, FieldAdd):
            res = z3.IntVal(0)
            for arg in e.args:
                res = self._mod(res + self._tr(arg))
            return res

        if isinstance(e, FieldMul):
            res = z3.IntVal(1)
            for arg in e.args:
                res = self._mod(res * self._tr(arg))
            return res

        if isinstance(e, FieldEq):
            return self._tr(e.left) == self._tr(e.right)

        if isinstance(e, FieldVar):
            return self.vars[e.name]

        if isinstance(e, FieldConst):
            if not (0 <= e.value < self.p):
                raise ValueError("constant outside field range")
            return z3.IntVal(e.value)

        raise TypeError(f"unknown AST node {type(e).__name__}")

# ---------------- tiny demo ------------------------------------------
def tiny_demo()->ParsedFormula:
    p = 17
    vars = {"x":"ff","m":"ff","z":"ff"}
    f1 = FieldEq(FieldAdd(FieldMul(FieldVar("m"), FieldVar("x")),
                          FieldConst(16), FieldVar("z")),
                 FieldConst(0))
    f2 = FieldEq(FieldMul(FieldVar("z"), FieldVar("x")), FieldConst(0))
    return ParsedFormula(p, vars, [f1, f2])

def run_demo()->None:
    s = FFIntSolver()
    res = s.check(tiny_demo())
    print("Result:", res)
    if res == z3.sat:
        print("Model:", s.model())

# ---------------------------------------------------------------------
def regress(dir_path:str)->None:
    for f in pathlib.Path(dir_path).rglob("*.smt2"):
        txt = f.read_text()
        expect = "unknown"
        if "; EXPECT: sat"   in txt: expect="sat"
        if "; EXPECT: unsat" in txt: expect="unsat"
        formula = tiny_demo()           # placeholder parser
        s = FFIntSolver()
        got = str(s.check(formula))
        ok  = (expect=="unknown") or (got==expect)
        print(f"{f.name:<30} expect={expect:7}  got={got:7}  {'✓' if ok else '✗'}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv)==2 and sys.argv[1]=="demo":
        run_demo()
    elif len(sys.argv)==3 and sys.argv[1]=="regress":
        regress(sys.argv[2])
    else:
        print("Usage:\n  ff_int_solver.py demo\n  ff_int_solver.py regress <dir>")
