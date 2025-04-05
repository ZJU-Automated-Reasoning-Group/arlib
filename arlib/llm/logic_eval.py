"""Evaluation script for LLM-based logical reasoning tasks.
Evaluates performance on:
- Satisfiability checking
- Abductive reasoning
- Quantifier elimination
- Interpolant generation
- Formula simplification
- Model counting
- Optimization
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import z3
import asyncio

from arlib.llm.llm4logic import LogicLLM, LLMConfig
from arlib.utils.logics import QF_LIA, QF_BV, QF_BOOL


@dataclass
class EvalResult:
    task_name: str
    correct: int
    total: int
    avg_time: float
    errors: List[str]


class LogicEvaluator:
    def __init__(self, llm: LogicLLM):
        self.llm = llm
        self.results: Dict[str, EvalResult] = {}

    async def evaluate_sat(self, benchmarks: List[Tuple[str, bool]]) -> EvalResult:
        """Evaluate satisfiability checking"""
        correct = 0
        errors = []
        times = []

        for formula, expected in benchmarks:
            try:
                start = time.time()
                result = await self.llm.check_sat(formula)
                elapsed = time.time() - start
                times.append(elapsed)

                if result == expected:
                    correct += 1
                else:
                    errors.append(f"Formula: {formula}, Expected: {expected}, Got: {result}")
            except Exception as e:
                errors.append(f"Error on {formula}: {str(e)}")

        return EvalResult(
            task_name="sat",
            correct=correct,
            total=len(benchmarks),
            avg_time=sum(times) / len(times) if times else 0,
            errors=errors
        )

    async def evaluate_interpolant(self, benchmarks: List[Tuple[str, str, str]]) -> EvalResult:
        """Evaluate interpolant generation"""
        correct = 0
        errors = []
        times = []

        for formula_a, formula_b, expected in benchmarks:
            try:
                start = time.time()
                result = await self.llm.compute_interpolant(formula_a, formula_b)
                elapsed = time.time() - start
                times.append(elapsed)

                # Verify interpolant properties using Z3
                solver = z3.Solver()
                a = z3.parse_smt2_string(formula_a)
                b = z3.parse_smt2_string(formula_b)
                i = z3.parse_smt2_string(result)

                # Check: A → I
                solver.push()
                solver.add(a)
                solver.add(z3.Not(i))
                valid_1 = solver.check() == z3.unsat
                solver.pop()

                # Check: I ∧ B is unsat
                solver.push()
                solver.add(i)
                solver.add(b)
                valid_2 = solver.check() == z3.unsat
                solver.pop()

                if valid_1 and valid_2:
                    correct += 1
                else:
                    errors.append(f"Invalid interpolant for A:{formula_a}, B:{formula_b}")

            except Exception as e:
                errors.append(f"Error on {formula_a}, {formula_b}: {str(e)}")

        return EvalResult(
            task_name="interpolant",
            correct=correct,
            total=len(benchmarks),
            avg_time=sum(times) / len(times) if times else 0,
            errors=errors
        )

    async def evaluate_qe(self, benchmarks: List[Tuple[str, str]]) -> EvalResult:
        """Evaluate quantifier elimination"""
        correct = 0
        errors = []
        times = []

        for formula, expected in benchmarks:
            try:
                start = time.time()
                result = await self.llm.eliminate_quantifiers(formula)
                elapsed = time.time() - start
                times.append(elapsed)

                # Verify equivalence using Z3
                solver = z3.Solver()
                f1 = z3.parse_smt2_string(formula)
                f2 = z3.parse_smt2_string(result)

                solver.add(z3.Not(z3.Equivalent(f1, f2)))
                if solver.check() == z3.unsat:
                    correct += 1
                else:
                    errors.append(f"Incorrect QE for {formula}")

            except Exception as e:
                errors.append(f"Error on {formula}: {str(e)}")

        return EvalResult(
            task_name="qe",
            correct=correct,
            total=len(benchmarks),
            avg_time=sum(times) / len(times) if times else 0,
            errors=errors
        )

    async def evaluate_all(self, benchmark_file: str) -> Dict[str, EvalResult]:
        """Run all evaluations"""
        # Load benchmarks from file
        benchmarks = self.load_benchmarks(benchmark_file)

        tasks = [
            self.evaluate_sat(benchmarks['sat']),
            self.evaluate_interpolant(benchmarks['interpolant']),
            self.evaluate_qe(benchmarks['qe'])
        ]

        results = await asyncio.gather(*tasks)
        return {r.task_name: r for r in results}

    def load_benchmarks(self, benchmark_file: str) -> Dict[str, List]:
        """Load benchmarks from file"""
        # TODO: Implement benchmark loading. For now, return some simple test cases
        return {
            'sat': [
                ("(declare-const x Int) (assert (and (> x 0) (< x 5)))", True),
                ("(declare-const x Int) (assert (and (> x 0) (< x 0)))", False)
            ],
            'interpolant': [
                ("(declare-const x Int) (assert (> x 0))",
                 "(declare-const x Int) (assert (< x -1))",
                 "(declare-const x Int) (assert (>= x 0))")
            ],
            'qe': [
                ("(exists ((x Int)) (and (> x 0) (< x 5)))",
                 "(declare-const x Int) (assert (and (> x 0) (< x 5)))")
            ]
        }

    def print_results(self):
        """Print evaluation results"""
        for task_name, result in self.results.items():
            print(f"\n=== {task_name.upper()} Evaluation ===")
            print(f"Accuracy: {result.correct}/{result.total} ({result.correct / result.total * 100:.2f}%)")
            print(f"Average Time: {result.avg_time:.3f}s")
            if result.errors:
                print("\nErrors:")
                for error in result.errors[:5]:  # Show first 5 errors
                    print(f"- {error}")


async def main():
    """Run evaluation"""
    config = LLMConfig(api_key="your-api-key")
    llm = LogicLLM(config)
    evaluator = LogicEvaluator(llm)

    results = await evaluator.evaluate_all("benchmarks/logic_benchmarks.smt2")
    evaluator.results = results
    evaluator.print_results()


if __name__ == "__main__":
    asyncio.run(main())
