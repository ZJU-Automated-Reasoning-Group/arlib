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


def _ensure_list(x: Union[Sequence, Z3Expr, SMTText]) -> List:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _parse_smt2_asserts(smt2_text: SMTText) -> List[Z3Expr]:
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    try:
        z3.parse_smt2_string(smt2_text, ctx=ctx)
    except z3.Z3Exception:
        # try to wrap if only a formula is given
        try:
            wrapped = f"(set-logic ALL)\n(assert {smt2_text})\n"
            z3.parse_smt2_string(wrapped, ctx=ctx)
        except z3.Z3Exception as e:
            raise e
    asts = []
    # Re-parse to collect assertions by reading into a temporary solver
    solver.add(z3.parse_smt2_string("(set-logic ALL)\n" + smt2_text, ctx=ctx))
    for a in solver.assertions():
        asts.append(a)
    return asts


def _to_asserts(formulas: Sequence[Union[SMTText, Z3Expr]]) -> List[Z3Expr]:
    out: List[Z3Expr] = []
    for f in formulas:
        if isinstance(f, z3.ExprRef):
            out.append(f)
        else:
            out.extend(_parse_smt2_asserts(f))
    return out


def _common_symbols(asA: List[Z3Expr], asB: List[Z3Expr]) -> List[str]:
    def symset(forms: List[Z3Expr]) -> set[str]:
        acc: set[str] = set()
        for e in forms:
            for sub in get_vars(e):
                acc.add(sub.decl().name())
        return acc

    return sorted(list(symset(asA) & symset(asB)))


def _mk_prompt(A: List[Z3Expr], B: List[Z3Expr]) -> str:
    A_text = "\n".join(f"(assert {e.sexpr()})" for e in A)
    B_text = "\n".join(f"(assert {e.sexpr()})" for e in B)
    commons = _common_symbols(A, B)
    guidelines = (
        "Return ONLY the interpolant as one SMT-LIB v2 S-expression. "
        "Do not include comments or explanations. "
        "Use only the shared symbols: " + ", ".join(commons)
    )
    return f"""
You are generating a Craig interpolant between two SMT-LIB sets A and B.

Requirements:
- The interpolant I must use only symbols common to A and B.
- A implies I must be valid. I and B together must be unsatisfiable.
- Keep it simple and quantifier-free if possible.

Provide only the single S-expression for I, no extra text.

Set A:
(set-logic ALL)
{A_text}

Set B:
(set-logic ALL)
{B_text}

{guidelines}
""".strip()


@dataclass
class InterpolantResult:
    interpolant: Optional[Z3Expr]
    valid_A_implies_I: bool
    unsat_I_and_B: bool
    raw_text: str


class _PromptInput(LLMToolInput):
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def __hash__(self) -> int:
        return hash(self.prompt)


class _PromptOutput(LLMToolOutput):
    def __init__(self, text: str) -> None:
        self.text = text


class _PromptTool(LLMTool):
    def _get_prompt(self, input: _PromptInput) -> str:
        return input.prompt

    def _parse_response(self, response: str, input: _PromptInput | None = None) -> _PromptOutput:
        return _PromptOutput(response)


class LLMInterpolantGenerator:
    def __init__(self, model_name: str = os.environ.get("ARLIB_LLM_MODEL", "gpt-4o-mini"), temperature: float = 0.2) -> None:
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
        A_list = _to_asserts(_ensure_list(A))
        B_list = _to_asserts(_ensure_list(B))
        prompt = _mk_prompt(A_list, B_list)

        last_text = ""
        for _ in range(max_attempts):
            out = self.tool.invoke(_PromptInput(prompt))
            text = out.text if out else ""
            last_text = text or last_text
            if not text:
                continue
            try:
                I = self._parse_interpolant(text)
            except Exception:
                continue
            v1, v2 = self._verify(A_list, B_list, I)
            if v1 and v2:
                return InterpolantResult(I, v1, v2, text)

        return InterpolantResult(None, False, False, last_text)

    @staticmethod
    def _parse_interpolant(text: str) -> Z3Expr:
        ctx = z3.Context()
        # Accept either a bare formula or wrapped assert
        try:
            exprs = z3.parse_smt2_string(f"(set-logic ALL)\n(assert {text})\n", ctx=ctx)
        except z3.Z3Exception:
            exprs = z3.parse_smt2_string(text, ctx=ctx)
        tmp = z3.Solver(ctx=ctx)
        tmp.add(exprs)
        asserts = tmp.assertions()
        if not asserts:
            raise ValueError("No formula parsed from LLM output")
        return asserts[0]

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
