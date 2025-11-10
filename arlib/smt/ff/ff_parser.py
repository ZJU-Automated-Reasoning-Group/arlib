#!/usr/bin/env python3
"""
ff_parser.py  –  Tiny SMT-LIB parser for the theory of Finite Fields
-------------------------------------------------------------------
Only the fragments that appear in QF_FF benchmarks are implemented.
If you meet a new construct the parser will raise  SyntaxError(...)
so you immediately see what to extend.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Union, Optional
import re, itertools, logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
#                            AST nodes
# ----------------------------------------------------------------------
@dataclass
class FieldExpr:  # abstract
    def collect_vars(self) -> Set[str]: raise NotImplementedError

@dataclass
class BoolExpr:   # abstract
    def collect_vars(self) -> Set[str]: raise NotImplementedError

# ----------  Boolean ---------------------------------------------------
@dataclass
class BoolVar(BoolExpr):
    name: str
    def collect_vars(self) -> Set[str]: return {self.name}

@dataclass
class BoolConst(BoolExpr):
    value: bool
    def collect_vars(self) -> Set[str]: return set()

@dataclass
class BoolNot(BoolExpr):
    arg: BoolExpr
    def collect_vars(self) -> Set[str]: return self.arg.collect_vars()

@dataclass
class BoolAnd(BoolExpr):
    args: List[BoolExpr]
    def collect_vars(self) -> Set[str]:
        return set().union(*(a.collect_vars() for a in self.args))

@dataclass
class BoolOr(BoolExpr):
    args: List[BoolExpr]
    def collect_vars(self) -> Set[str]:
        return set().union(*(a.collect_vars() for a in self.args))

@dataclass
class BoolImplies(BoolExpr):
    left: BoolExpr
    right: BoolExpr
    def collect_vars(self) -> Set[str]:
        return self.left.collect_vars() | self.right.collect_vars()

# ----------  Field -----------------------------------------------------
@dataclass
class FieldVar(FieldExpr):
    name: str
    def collect_vars(self) -> Set[str]: return {self.name}

@dataclass
class FieldConst(FieldExpr):
    value: int                 # canonical repr 0 … p-1
    def collect_vars(self) -> Set[str]: return set()

@dataclass
class FieldAdd(FieldExpr):
    args: List[FieldExpr]
    def collect_vars(self) -> Set[str]:
        return set().union(*(a.collect_vars() for a in self.args))

@dataclass
class FieldMul(FieldExpr):
    args: List[FieldExpr]
    def collect_vars(self) -> Set[str]:
        return set().union(*(a.collect_vars() for a in self.args))

@dataclass
class FieldEq(BoolExpr):       # equality is Boolean!
    left: FieldExpr
    right: FieldExpr
    def collect_vars(self) -> Set[str]:
        return self.left.collect_vars() | self.right.collect_vars()

@dataclass
class FieldITE(FieldExpr):
    cond: BoolExpr
    then_e: FieldExpr
    else_e: FieldExpr
    def collect_vars(self)->Set[str]:
        return self.cond.collect_vars()|self.then_e.collect_vars()|self.else_e.collect_vars()

# ----------------------------------------------------------------------
@dataclass
class ParsedFormula:
    field_size: int
    variables : Dict[str, int]       # name → p   (one field only)
    ff_assertions   : List[FieldExpr]
    bool_assertions : List[BoolExpr]

# ======================================================================
class FFParser:
    _FF_LITERAL_RE = re.compile(r"#f(\d+)m(\d+)$")   # #f16m17

    # ------------- tokenisation --------------------------------------
    def _tokenise(self, text:str) -> List[str]:
        # strip comments
        text = re.sub(r";[^\n]*", "", text)
        # add spaces around parentheses
        text = text.replace("(", " ( ").replace(")", " ) ")
        return text.split()

    # ------------- simple S-expression reader ------------------------
    def _read_sexp(self, toks:List[str], pos:int=0) -> Tuple[Union[str,List],int]:
        if pos>=len(toks):
            raise SyntaxError("unexpected EOF")
        tok = toks[pos]
        if tok == '(':
            lst = []
            pos += 1
            while pos < len(toks) and toks[pos] != ')':
                node, pos = self._read_sexp(toks, pos)
                lst.append(node)
            if pos >= len(toks):
                raise SyntaxError("missing ')'")
            return lst, pos+1
        if tok == ')':
            raise SyntaxError("unexpected ')'")
        return tok, pos+1

    # ------------- public entry point --------------------------------

    # --- inside class FFParser  -------------------------------------------

    def parse_formula(self, smt: str) -> ParsedFormula:
        toks = self._tokenise(smt)
        sexps, pos = [], 0
        while pos < len(toks):
            s, pos = self._read_sexp(toks, pos)
            sexps.append(s)

        field_size: Optional[int] = None
        variables : Dict[str, int] = {}
        ff_asserts: List[FieldExpr]  = []
        bool_asserts: List[BoolExpr] = []

        # commands we ignore completely -------------------------------↓
        SKIP_CMDS = {
            'set-logic', 'set-info', 'check-sat', 'check-sat-assuming',
            'exit', 'get-model', 'get-value', 'get-proof', 'get-unsat-core'
        }

        # pass 1 – gather variable and field information --------------
        for s in sexps:
            if not (isinstance(s, list) and s):
                continue
            head = s[0]
            if head in SKIP_CMDS:
                continue
            if head in ('declare-fun', 'declare-const'):
                name = s[1]
                sort = s[-1]
                if sort == 'Bool':
                    continue
                if isinstance(sort, list) and sort[:2] == ['_', 'FiniteField']:
                    p = int(sort[2])
                    field_size = field_size or p
                    if p != field_size:
                        raise SyntaxError("several different fields not supported")
                    variables[name] = p
                else:
                    raise SyntaxError(f"unknown sort {sort}")

        # literal-only field size fallback ----------------------------
        if field_size is None:
            m = self._FF_LITERAL_RE.search(smt)
            if m:
                field_size = int(m.group(2))
        if field_size is None:
            raise SyntaxError("could not determine field size")

        # pass 2 – translate assertions ------------------------------
        for s in sexps:
            if not (isinstance(s, list) and s):
                continue
            head = s[0]
            if head in SKIP_CMDS or head.startswith('set-'):
                continue
            if head == 'assert':
                bool_asserts.append(self._to_bool(s[1], field_size, variables))
            elif head in ('declare-fun', 'declare-const', 'define-sort'):
                continue
            else:
                raise SyntaxError(f"unsupported top-level command {head}")

        return ParsedFormula(field_size, variables, ff_asserts, bool_asserts)

    # ---------------- expression translation -------------------------
    def _to_bool(self, e, p:int, vars:Dict[str,int]) -> BoolExpr:
        if isinstance(e,str):
            if e in ('true','false'):
                return BoolConst(e=='true')
            # Boolean variable?
            return BoolVar(e)

        # list
        head, *args = e
        if head == 'not':
            return BoolNot(self._to_bool(args[0],p,vars))
        if head == 'and':
            return BoolAnd([self._to_bool(a,p,vars) for a in args])
        if head == 'or':
            return BoolOr([self._to_bool(a,p,vars) for a in args])
        if head == '=>':
            a1,a2 = args
            return BoolImplies(self._to_bool(a1,p,vars),
                                self._to_bool(a2,p,vars))
        if head == '=':            # (= <field> <field>)
            if len(args)!=2:
                raise SyntaxError("= arity")
            return FieldEq(self._to_field(args[0],p,vars),
                           self._to_field(args[1],p,vars))
        raise SyntaxError(f"unknown boolean op {head}")

    def _to_field(self, e, p:int, vars:Dict[str,int]) -> FieldExpr:
        if isinstance(e,str):
            # variable?
            if e in vars: return FieldVar(e)
            # finite-field literal  #fXmP
            m = self._FF_LITERAL_RE.fullmatch(e)
            if m:
                val, mod = int(m.group(1)), int(m.group(2))
                if mod != p:
                    raise SyntaxError("mixing different fields")
                return FieldConst(val % p)
            raise SyntaxError(f"unbound symbol {e}")

        head,*args = e
        if head == 'ff.add':
            return FieldAdd([self._to_field(a,p,vars) for a in args])
        if head == 'ff.mul':
            return FieldMul([self._to_field(a,p,vars) for a in args])
        if head == 'ite':          # (ite <bool> <field> <field>)
            b, t, f = args
            return FieldITE(self._to_bool(b,p,vars),
                            self._to_field(t,p,vars),
                            self._to_field(f,p,vars))
        raise SyntaxError(f"unknown field op {head}")

# ----------------------------------------------------------------------
#                           quick self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE = """
    (set-logic QF_FF)
    ; simple example
    (declare-fun x () (_ FiniteField 17))
    (declare-fun y () (_ FiniteField 17))
    (declare-fun b () Bool)
    (assert (=> (and (= x #f1m17) b)
                (or (= y (ff.add x #f16m17))
                    (= y (ff.mul x x)))))
    (check-sat)
    """

    pf = FFParser().parse_formula(SAMPLE)
    print("Field size :", pf.field_size)
    print("Variables  :", list(pf.variables.keys()))
    print("#Bool asserts:", len(pf.bool_assertions))
    print("AST         :", pf.bool_assertions[0])
