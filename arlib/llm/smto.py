"""
Orax: An SMTO solver using LLM as oracle handler

Recent advancements in SMT solving have driven progress in formula-based software verification, but they face significant challenges when analyzing open programs with components lacking formal specifications, such as third-party libraries or deep learning models. This challenge has led to the emergence of the Satisfiability Modulo Theories and Oracles (SMTO) problem, where black-box components can be accessed as oracles with observable inputs and outputs but unknown implementations. Current approaches like Delphi and Saadhak attempt to solve SMTO problems by combining conventional SMT solvers with oracle handlers, but they struggle to effectively integrate oracle mapping information with SMT reasoning capabilities. 

To address these limitations, we propose to establish an oracle mapping information feedback loop between the SMT solver and an oracle handler powered by a large language model (LLM).

Next, we brainstorm some ideas on how to combine LLM with existing SMT solvers.

TODO: how about using LLM to analzying the "backbox" components directly? E.g.,
1. the documentation of the component (if available)
2. the binary code (possibly obfuscated) of the component (if available)
3. other "implicit" knowledge about the component  (e.g., online discussions about the component)
..

What to guess/infer/learn from the above information?
1. the "specification" of the component, i.e., the "contract" that the component adheres to
2. the possible "internal" implementation of the component
3. ...

Another direction is to use LLM to perform some logcial reasoning that might go beyond the capability of existing SMT solvers, e.g.,
1. abductive inference (e.g., from observations to possible hypotheses)over complex constraints that cannot be encoded via SMT theories (at least a subset of them cannot be ...)
2. ...


"""

from typing import Dict, List, Optional, Union, Callable
import z3
from openai import OpenAI
import time
from dataclasses import dataclass


@dataclass
class OracleInfo:
    """Information about an oracle function"""
    name: str
    input_types: List[z3.SortRef]
    output_type: z3.SortRef
    description: str
    examples: List[Dict]  # List of input-output examples
    # Should we allow for orcle that is a function body? i.e., the oracle is a function body that can be used as a constraint
    # Besides, for "static analysis", the oracle could also be some form of  "summary" or "transfer functions"


class OraxSolver:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize Orax solver
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.oracles: Dict[str, OracleInfo] = {}
        self.solver = z3.Solver()
        self.cache: Dict[str, Dict] = {}  # Cache for oracle results

    def register_oracle(self, oracle_info: OracleInfo):
        """Register an oracle function"""
        self.oracles[oracle_info.name] = oracle_info

    def query_llm(self, oracle: OracleInfo, inputs: Dict) -> Optional[Union[int, float, bool, str]]:
        """Query LLM to simulate oracle function"""
        # Construct prompt with oracle description and examples
        prompt = f"Act as the following function:\n{oracle.description}\n\nExamples:\n"
        for example in oracle.examples:
            prompt += f"Input: {example['input']}\nOutput: {example['output']}\n"

        prompt += f"\nNow, given the input: {inputs}\nWhat is the output?"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise function evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            return self._parse_llm_response(response.choices[0].message.content, oracle.output_type)
        except Exception as e:
            print(f"LLM query failed: {e}")
            return None

    def _parse_llm_response(self, response: str, output_type: z3.SortRef) -> Optional[Union[int, float, bool, str]]:
        """Parse LLM response according to expected output type"""
        try:
            if output_type == z3.BoolSort():
                return response.lower().strip() in ['true', '1', 'yes']
            elif output_type == z3.IntSort():
                return int(response.strip())
            elif output_type == z3.RealSort():
                return float(response.strip())
            else:
                return response.strip()
        except:
            return None

    def add_constraint(self, constraint: z3.BoolRef):
        """Add constraint to the solver"""
        self.solver.add(constraint)

    def check(self) -> Optional[z3.ModelRef]:
        """
        Check satisfiability with oracle feedback loop
        Returns model if satisfiable, None if unsatisfiable
        """
        max_iterations = 10
        for iteration in range(max_iterations):
            result = self.solver.check()
            if result == z3.unsat:
                return None

            model = self.solver.model()
            is_valid = self._validate_model_with_oracles(model)

            if is_valid:
                return model

            # Add learned constraints from oracle feedback
            self._add_oracle_constraints(model)

        return None

    def _validate_model_with_oracles(self, model: z3.ModelRef) -> bool:
        """Validate model by checking oracle constraints"""
        for oracle_name, oracle_info in self.oracles.items():
            # Find all applications of this oracle in the model
            for decl in model.decls():
                if decl.name() == oracle_name:
                    args = [model.eval(arg, True) for arg in decl.children()]
                    expected = model.eval(decl(), True)

                    # Query LLM for actual result
                    inputs = {f"arg{i}": arg for i, arg in enumerate(args)}
                    actual = self.query_llm(oracle_info, inputs)

                    if actual is None or actual != expected:
                        return False
        return True

    def _add_oracle_constraints(self, model: z3.ModelRef):
        """Add learned constraints from oracle feedback"""
        for oracle_name, oracle_info in self.oracles.items():
            for decl in model.decls():
                if decl.name() == oracle_name:
                    args = [model.eval(arg, True) for arg in decl.children()]
                    inputs = {f"arg{i}": arg for i, arg in enumerate(args)}

                    # Check cache first
                    cache_key = f"{oracle_name}_{inputs}"
                    if cache_key not in self.cache:
                        self.cache[cache_key] = self.query_llm(oracle_info, inputs)

                    actual = self.cache[cache_key]
                    if actual is not None:
                        # Add constraint that this oracle application must equal actual result
                        constraint = decl() == actual
                        self.solver.add(constraint)


def example_usage():
    """Example usage of OraxSolver"""
    # Initialize solver
    solver = OraxSolver("your-api-key")

    # Register an oracle for string length
    strlen_oracle = OracleInfo(
        name="strlen",
        input_types=[z3.StringSort()],
        output_type=z3.IntSort(),
        description="Calculate the length of a string",
        examples=[
            {"input": {"arg0": "hello"}, "output": "5"},
            {"input": {"arg0": "world!"}, "output": "6"}
        ]
    )
    solver.register_oracle(strlen_oracle)

    # Create variables and constraints
    s = z3.String('s')
    length = z3.Int('length')

    # Add constraints
    solver.add_constraint(z3.Length(s) == length)
    solver.add_constraint(length > 5)
    solver.add_constraint(length < 10)

    # Solve
    model = solver.check()
    if model:
        print(f"Solution found: s = {model[s]}, length = {model[length]}")
    else:
        print("No solution found")


if __name__ == "__main__":
    example_usage()
