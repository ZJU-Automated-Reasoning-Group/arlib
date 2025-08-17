"""Tiny demo for the LLM interpolant PoC.

Run:
  ARLIB_LLM_MODEL=gpt-4o-mini python -m arlib.llm.interpolant.demo
"""

from arlib.llm.interpolant.llm_interpolant import LLMInterpolantGenerator


def main():
    # Example over LIA
    A = ["(declare-fun x () Int)", "(declare-fun y () Int)", "(assert (> x 6))", "(assert (= y (+ x 1)))"]
    B = ["(declare-fun y () Int)", "(assert (<= y 4))"]

    gen = LLMInterpolantGenerator()
    res = gen.generate(A, B)
    print("Interpolant:", res.raw_text)
    print("A => I:", res.valid_A_implies_I, "; I & B unsat:", res.unsat_I_and_B)


if __name__ == "__main__":
    main()
