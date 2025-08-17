from __future__ import annotations

"""LLM-based Craig interpolant generation PoC.

Workflow:
1) Take two sets of assertions A and B, each in SMT-LIB (strings) or z3 ASTs.
2) Build a concise prompt asking the LLM to return an interpolant I in SMT-LIB
   restricted to symbols common to A and B.
3) Parse the returned candidate with z3, and verify:
   - A => I is valid
   - I ∧ B is unsat
   If both hold, accept; otherwise, report failure.

Notes:
- This is a PoC: minimal guards, no complex few-shoting.
- The LLM backend is reused from arlib.llm.abduct.base.LLMViaTool.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence, Union

import os
import z3
from z3.z3util import get_vars

from arlib.llm.llmtool.LLM_tool import LLMTool, LLMToolInput, LLMToolOutput
from arlib.llm.llmtool.logger import Logger


SMTText = str
Z3Expr = z3.ExprRef


def _parse_smt2_asserts(smt2_text: SMTText) -> List[Z3Expr]:
    solver = z3.Solver()
    for attempt in [smt2_text, f"(assert {smt2_text})\n"]:
        try:
            parsed = z3.parse_smt2_string(f"(set-logic ALL)\n{attempt}")
            for expr in parsed:
                solver.add(expr)
            return list(solver.assertions())
        except z3.Z3Exception:
            continue
    raise z3.Z3Exception("Failed to parse SMT2 text")


def _to_asserts(formulas: Union[Sequence[Union[SMTText, Z3Expr]], SMTText, Z3Expr]) -> List[Z3Expr]:
    if not isinstance(formulas, (list, tuple)):
        formulas = [formulas]

    z3_exprs = [f for f in formulas if isinstance(f, z3.ExprRef)]
    smt2_texts = [f for f in formulas if not isinstance(f, z3.ExprRef)]

    return z3_exprs + (_parse_smt2_asserts("\n".join(smt2_texts)) if smt2_texts else [])


def _common_symbols(asA: List[Z3Expr], asB: List[Z3Expr]) -> List[str]:
    get_names = lambda forms: {v.decl().name() for e in forms for v in get_vars(e)}
    return sorted(get_names(asA) & get_names(asB))


def _mk_prompt(A: List[Z3Expr], B: List[Z3Expr]) -> str:
    A_text = "\n".join(f"(assert {e.sexpr()})" for e in A)
    B_text = "\n".join(f"(assert {e.sexpr()})" for e in B)
    commons = ", ".join(_common_symbols(A, B))
    return f"""Generate a Craig interpolant I between sets A and B.
Requirements: A⟹I valid, I∧B unsat, use only shared symbols: {commons}
Return ONLY the S-expression, no explanations.

A: (set-logic ALL)\n{A_text}
B: (set-logic ALL)\n{B_text}"""


@dataclass
class InterpolantResult:
    interpolant: Optional[Z3Expr]
    valid_A_implies_I: bool
    unsat_I_and_B: bool
    raw_text: str


class _PromptTool(LLMTool):
    def _get_prompt(self, input: str) -> str:
        return input

    def _parse_response(self, response: str, input: str | None = None) -> str:
        return response


class LLMInterpolantGenerator:
    def __init__(self, model_name: str = os.environ.get("ARLIB_LLM_MODEL", "glm-4-flash"), temperature: float = 0.2) -> None:
        log_dir = os.environ.get("ARLIB_LOG_DIR", ".arlib_logs")
        os.makedirs(log_dir, exist_ok=True)
        logger = Logger(os.path.join(log_dir, "interpolant_llm.log"))
        self.tool = _PromptTool(
            model_name=model_name,
            temperature=temperature,
            language="en",
            max_query_num=3,
            logger=logger,
        )

    def generate(
        self,
        A: Union[Sequence[Union[SMTText, Z3Expr]], SMTText, Z3Expr],
        B: Union[Sequence[Union[SMTText, Z3Expr]], SMTText, Z3Expr],
        max_attempts: int = 3,
    ) -> InterpolantResult:
        A_list, B_list = _to_asserts(A), _to_asserts(B)
        prompt = _mk_prompt(A_list, B_list)

        last_text = ""
        for _ in range(max_attempts):
            text = self.tool.invoke(prompt) or ""
            last_text = text or last_text
            if not text:
                continue
            try:
                I = self._parse_interpolant(text)
                v1, v2 = self._verify(A_list, B_list, I)
                if v1 and v2:
                    return InterpolantResult(I, v1, v2, text)
            except Exception:
                continue

        return InterpolantResult(None, False, False, last_text)

    @staticmethod
    def _parse_interpolant(text: str) -> Z3Expr:
        for attempt in [f"(assert {text})\n", text]:
            try:
                exprs = z3.parse_smt2_string(f"(set-logic ALL)\n{attempt}")
                tmp = z3.Solver()
                tmp.add(exprs)
                if tmp.assertions():
                    return tmp.assertions()[0]
            except z3.Z3Exception:
                continue
        raise ValueError("No valid formula parsed from LLM output")

    @staticmethod
    def _verify(A: List[Z3Expr], B: List[Z3Expr], I: Z3Expr) -> Tuple[bool, bool]:
        # Check A => I by testing A ∧ ¬I is unsat
        s1 = z3.Solver()
        s1.add(A)
        s1.add(z3.Not(I))
        v1 = s1.check() == z3.unsat

        # Check I ∧ B is unsat
        s2 = z3.Solver()
        s2.add(I)
        s2.add(B)
        v2 = s2.check() == z3.unsat
        return v1, v2
