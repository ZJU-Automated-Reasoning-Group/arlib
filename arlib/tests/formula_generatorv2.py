"""
FIXME: sort missmatch problem for bit-vectors..
"""
import random
import z3
from typing import List, Optional, Union, Tuple


class SMTFuzzer:
    """Enhanced formula generator supporting various SMT theories"""

    def __init__(self, logic: str = "ALL",
                 min_depth: int = 3,
                 max_depth: int = 8,
                 max_terms: int = 30):
        self.logic = logic
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_terms = max_terms

        # Theory-specific variables
        self.arrays: List[z3.ArrayRef] = []
        self.ints: List[z3.ArithRef] = []
        self.reals: List[z3.ArithRef] = []
        self.bvs: List[z3.BitVecRef] = []
        self.array_sorts = []
        self.strings: List[z3.SeqRef] = []
        self.fp_nums: List[z3.FPRef] = []
        self.fp_sorts = [z3.Float16(), z3.Float32(), z3.Float64()]

        # Initialize variables based on logic
        if any(x in logic for x in ["LIA", "AUFLIA"]) or logic == "ALL":
            for i in range(5):
                self.ints.append(z3.Int(f'x_{i}'))

        if "BV" in logic or logic == "ALL":
            for i in range(5):
                width = random.choice([8, 16, 32, 64])
                self.bvs.append(z3.BitVec(f'bv_{i}', width))

    def generate_bv_expr(self, depth: int = 0, width: Optional[int] = None) -> z3.BitVecRef:
        """Generate bit-vector expressions with matching widths"""
        # Set default width if none provided
        if width is None and self.bvs:
            width = random.choice(self.bvs).size()
        elif width is None:
            width = random.choice([8, 16, 32, 64])

        # Base case: return variable or constant
        if depth >= self.max_depth or random.random() < 0.3:
            matching_bvs = [bv for bv in self.bvs if bv.size() == width]
            if matching_bvs:
                return random.choice(matching_bvs)
            # Create new BV with correct width
            new_bv = z3.BitVec(f'bv_new_{len(self.bvs)}', width)
            self.bvs.append(new_bv)
            return new_bv

        # Generate expression with consistent width
        ops = [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x & y,
            lambda x, y: x | y,
            lambda x, y: x ^ y,
            lambda x: ~x,
            lambda x, y: z3.ULT(x, y),
            lambda x, y: z3.UGT(x, y),
            lambda x: z3.RotateLeft(x, random.randint(0, width - 1)),
            lambda x: z3.RotateRight(x, random.randint(0, width - 1))
        ]

        op = random.choice(ops)

        if op.__code__.co_argcount == 1:
            expr = self.generate_bv_expr(depth + 1, width)
            return op(expr)
        else:
            return op(self.generate_bv_expr(depth + 1, width),
                      self.generate_bv_expr(depth + 1, width))

    def generate_formula(self) -> z3.ExprRef:
        """Generate a random formula combining multiple theories"""
        constraints = []

        # Generate BV constraints
        if "BV" in self.logic or self.logic == "ALL":
            for _ in range(random.randint(1, self.max_terms)):
                expr = self.generate_bv_expr()
                if z3.is_bool(expr):
                    constraints.append(expr)
                else:
                    op = random.choice(['<', '<=', '>', '>=', '==', '!='])
                    if op == '<':
                        constraints.append(z3.ULT(expr, self.generate_bv_expr()))
                    elif op == '<=':
                        constraints.append(z3.ULE(expr, self.generate_bv_expr()))
                    elif op == '>':
                        constraints.append(z3.UGT(expr, self.generate_bv_expr()))
                    elif op == '>=':
                        constraints.append(z3.UGE(expr, self.generate_bv_expr()))
                    else:
                        constraints.append(expr == self.generate_bv_expr())

    def generate_smt2(self) -> str:
        """Generate SMT-LIB2 string representation"""
        formula = self.generate_formula()
        solver = z3.Solver()

        # Set logic based on theories used
        if "AUFLIA" in self.logic:
            solver.set(logic="QF_AUFLIA")
        elif "AUFBV" in self.logic:
            solver.set(logic="QF_AUFBV")
        elif "BV" in self.logic:
            solver.set(logic="QF_BV")
        elif "LIA" in self.logic:
            solver.set(logic="QF_LIA")
        elif "ARRAY" in self.logic:
            solver.set(logic="QF_AX")
        elif self.logic == "ALL":
            solver.set(auto_config=True)

        solver.add(formula)
        return solver.to_smt2()

    def _init_operations(self):
        """Initialize theory-specific operations"""
        self.array_ops = [
            lambda a, i: z3.Select(a, i),
            lambda a, i, v: z3.Store(a, i, v)
        ]

        self.string_ops = [
            lambda s1, s2: z3.Concat(s1, s2),
            lambda s: z3.Length(s),
            lambda s1, s2: z3.Contains(s1, s2),
            lambda s1, s2: z3.PrefixOf(s1, s2),
            lambda s1, s2: z3.SuffixOf(s1, s2)
        ]

        self.fp_ops = [
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y,
            lambda x: z3.fpAbs(x),
            lambda x: z3.fpNeg(x),
            lambda x: z3.fpIsInf(x),
            lambda x: z3.fpIsNaN(x)
        ]

    def generate_array(self, depth: int = 0) -> z3.ArrayRef:
        """Generate array expressions"""
        if not self.arrays or depth >= self.max_depth:
            # Create new array
            index_sort = random.choice([z3.IntSort(), z3.BitVecSort(32)])
            value_sort = random.choice([z3.IntSort(), z3.BitVecSort(32), z3.BoolSort()])
            arr = z3.Array('arr_%d' % len(self.arrays), index_sort, value_sort)
            self.arrays.append(arr)
            return arr

        arr = random.choice(self.arrays)
        if random.random() < 0.3:
            return arr

        op = random.choice(self.array_ops)
        if op.__code__.co_argcount == 2:
            index = self._generate_index(arr.domain())
            return op(arr, index)
        else:
            index = self._generate_index(arr.domain())
            value = self._generate_value(arr.range())
            return op(arr, index, value)

    def generate_string(self, depth: int = 0) -> z3.SeqRef:
        """Generate string expressions"""
        if not self.strings or depth >= self.max_depth:
            # Create new string variable
            s = z3.String('str_%d' % len(self.strings))
            self.strings.append(s)
            return s

        if random.random() < 0.3:
            return random.choice(self.strings)

        op = random.choice(self.string_ops)
        if op.__code__.co_argcount == 1:
            return op(self.generate_string(depth + 1))
        else:
            return op(self.generate_string(depth + 1),
                      self.generate_string(depth + 1))

    def generate_fp(self, depth: int = 0) -> z3.FPRef:
        """Generate floating-point expressions"""
        if not self.fp_nums or depth >= self.max_depth:
            sort = random.choice(self.fp_sorts)
            fp = z3.FP('fp_%d' % len(self.fp_nums), sort)
            self.fp_nums.append(fp)
            return fp

        if random.random() < 0.3:
            return random.choice(self.fp_nums)

        op = random.choice(self.fp_ops)
        if op.__code__.co_argcount == 1:
            return op(self.generate_fp(depth + 1))
        else:
            return op(self.generate_fp(depth + 1),
                      self.generate_fp(depth + 1))

    def _generate_index(self, sort):
        """Generate index value for arrays"""
        if sort == z3.IntSort():
            return z3.Int('i_%d' % random.randint(0, 1000))
        elif sort == z3.BitVecSort(32):
            return z3.BitVec('bv_%d' % random.randint(0, 1000), 32)
        else:
            raise ValueError(f"Unsupported index sort: {sort}")

    def _generate_value(self, sort):
        """Generate value for arrays"""
        if sort == z3.IntSort():
            return z3.IntVal(random.randint(-100, 100))
        elif sort == z3.BitVecSort(32):
            return z3.BitVecVal(random.randint(0, 2 ** 32 - 1), 32)
        elif sort == z3.BoolSort():
            return z3.BoolVal(random.choice([True, False]))
        else:
            raise ValueError(f"Unsupported value sort: {sort}")


def demo():
    """Demo usage of SMTFuzzer"""
    fuzzer = SMTFuzzer(logic="QF_BV", max_depth=5, max_terms=10)
    smt2_str = fuzzer.generate_smt2()
    print(smt2_str)

    # Try to solve the generated formula
    solver = z3.Solver()
    solver.from_string(smt2_str)
    print("Result:", solver.check())


if __name__ == "__main__":
    demo()
