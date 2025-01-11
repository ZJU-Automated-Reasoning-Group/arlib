from dataclasses import dataclass
from typing import Dict, Any, List, Union, Optional, Set
from abc import ABC, abstractmethod
from z3 import *
import logging
from enum import Enum
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FieldError(Exception):
    """Base class for finite field related errors."""
    pass


class InvalidFieldSize(FieldError):
    """Raised when field size is invalid."""
    pass


class InvalidFieldOperation(FieldError):
    """Raised when field operation is invalid."""
    pass


class OperationType(Enum):
    ADD = "add"
    MUL = "mul"
    NEG = "neg"
    SUB = "sub"


@dataclass
class FieldExpr(ABC):
    """Abstract base class for field expressions."""

    @abstractmethod
    def to_z3(self, ctx) -> BitVecRef:
        """Convert expression to Z3 formula."""
        pass

    @abstractmethod
    def collect_vars(self) -> Set[str]:
        """Collect all variable names used in expression."""
        pass

    def __add__(self, other):
        return FieldAdd([self, other])

    def __mul__(self, other):
        return FieldMul([self, other])


@dataclass
class BoolExpr(ABC):
    """Abstract base class for boolean expressions."""

    @abstractmethod
    def to_z3(self, ctx) -> BoolRef:
        pass

    @abstractmethod
    def collect_vars(self) -> Set[str]:
        pass


@dataclass
class BoolVar(BoolExpr):
    """Boolean variable."""
    name: str

    def to_z3(self, ctx) -> BoolRef:
        return Bool(self.name)

    def collect_vars(self) -> Set[str]:
        return {self.name}


@dataclass
class BoolImplies(BoolExpr):
    """Boolean implication."""
    left: BoolExpr
    right: BoolExpr

    def to_z3(self, ctx) -> BoolRef:
        return Implies(self.left.to_z3(ctx), self.right.to_z3(ctx))

    def collect_vars(self) -> Set[str]:
        return self.left.collect_vars().union(self.right.collect_vars())


@dataclass
class BoolOr(BoolExpr):
    """Boolean OR operation."""
    args: List[BoolExpr]

    def to_z3(self, ctx) -> BoolRef:
        return Or(*[arg.to_z3(ctx) for arg in self.args])

    def collect_vars(self) -> Set[str]:
        vars = set()
        for arg in self.args:
            vars.update(arg.collect_vars())
        return vars


@dataclass
class FieldITE(FieldExpr):
    """If-then-else expression returning field value."""
    condition: BoolExpr
    then_expr: FieldExpr
    else_expr: FieldExpr

    def to_z3(self, ctx) -> BitVecRef:
        return If(self.condition.to_z3(ctx),
                  self.then_expr.to_z3(ctx),
                  self.else_expr.to_z3(ctx))

    def collect_vars(self) -> Set[str]:
        return (self.condition.collect_vars()
                .union(self.then_expr.collect_vars())
                .union(self.else_expr.collect_vars()))


@dataclass
class FieldVar(FieldExpr):
    """Represents a field variable."""
    name: str

    def to_z3(self, ctx) -> BitVecRef:
        sort = ctx.sorts[ctx.current_field]
        var = BitVec(self.name, sort.size())
        # Add range constraint
        ctx.solver.add(ULT(var, BitVecVal(ctx.field_size, sort.size())))
        return var

    def collect_vars(self) -> Set[str]:
        return {self.name}


@dataclass
class FieldConst(FieldExpr):
    """Represents a field constant."""
    value: int

    def to_z3(self, ctx) -> BitVecRef:
        sort = ctx.sorts[ctx.current_field]
        if not 0 <= self.value < ctx.field_size:
            raise InvalidFieldOperation(f"Constant {self.value} outside field range")
        return BitVecVal(self.value, sort.size())

    def collect_vars(self) -> Set[str]:
        return set()


@dataclass
class FieldOp(FieldExpr):
    """Base class for field operations."""
    args: List[FieldExpr]

    @abstractmethod
    def op_type(self) -> OperationType:
        pass

    def collect_vars(self) -> Set[str]:
        vars = set()
        for arg in self.args:
            vars.update(arg.collect_vars())
        return vars


@dataclass
class FieldAdd(FieldOp):
    """Represents field addition."""

    def op_type(self) -> OperationType:
        return OperationType.ADD

    def to_z3(self, ctx) -> BitVecRef:
        if not self.args:
            return BitVecVal(0, ctx.sorts[ctx.current_field].size())

        result = self.args[0].to_z3(ctx)
        for arg in self.args[1:]:
            result = URem(result + arg.to_z3(ctx),
                          BitVecVal(ctx.field_size, result.size()))
        return result


@dataclass
class FieldAnd(FieldExpr):
    """Represents boolean AND operation."""
    args: List[FieldExpr]

    def to_z3(self, ctx) -> BoolRef:
        return And(*[arg.to_z3(ctx) for arg in self.args])

    def collect_vars(self) -> Set[str]:
        vars = set()
        for arg in self.args:
            vars.update(arg.collect_vars())
        return vars


@dataclass
class FieldNot(FieldExpr):
    """Represents boolean NOT operation."""
    arg: FieldExpr

    def to_z3(self, ctx) -> BoolRef:
        return Not(self.arg.to_z3(ctx))

    def collect_vars(self) -> Set[str]:
        return self.arg.collect_vars()


@dataclass
class FieldMul(FieldOp):
    """Represents field multiplication."""

    def op_type(self) -> OperationType:
        return OperationType.MUL

    def to_z3(self, ctx) -> BitVecRef:
        if not self.args:
            return BitVecVal(1, ctx.sorts[ctx.current_field].size())

        result = self.args[0].to_z3(ctx)
        for arg in self.args[1:]:
            result = URem(result * arg.to_z3(ctx),
                          BitVecVal(ctx.field_size, result.size()))
        return result


@dataclass
class FieldEq(FieldExpr):
    """Represents field equality."""
    left: FieldExpr
    right: FieldExpr

    def to_z3(self, ctx) -> BoolRef:
        return self.left.to_z3(ctx) == self.right.to_z3(ctx)

    def collect_vars(self) -> Set[str]:
        return self.left.collect_vars().union(self.right.collect_vars())


@dataclass
class LetBinding:
    """Represents a let binding."""
    var_name: str
    value: FieldExpr


@dataclass
class ExtendedSMTContext:
    """Enhanced SMT context with IR support."""
    solver: Solver
    variables: Dict[str, FieldVar]
    sorts: Dict[str, BitVecSort]
    field_size: Optional[int] = None
    current_field: Optional[str] = None

    def add_variable(self, name: str, sort: str):
        """Add a new variable to the context."""
        if name in self.variables:
            raise FieldError(f"Variable {name} already declared")
        if sort == 'Bool':
            var = BoolVar(name)
        else:
            var = FieldVar(name)
        self.variables[name] = var
        return var


class EnhancedSMTParser:
    """Enhanced SMT parser with IR support."""

    def __init__(self):
        self.ctx = ExtendedSMTContext(
            solver=Solver(),
            variables={},
            sorts={},
        )

    def tokenize(self, text: str) -> List[str]:
        """Tokenize SMT-LIB input."""
        # Remove comments
        text = re.sub(';.*\n', '\n', text)
        # Add spaces around parentheses
        text = text.replace('(', ' ( ').replace(')', ' ) ')
        # Split into tokens
        return [token for token in text.split() if token.strip()]

    def parse_sexp(self, tokens: List[str]) -> List:
        """Parse s-expressions into nested lists."""
        if not tokens:
            return []

        token = tokens.pop(0)
        if token == '(':
            lst = []
            while tokens and tokens[0] != ')':
                lst.append(self.parse_sexp(tokens))
            if not tokens:
                raise SyntaxError("Unexpected EOF")
            tokens.pop(0)  # Remove ')'
            return lst
        elif token == ')':
            raise SyntaxError("Unexpected closing parenthesis")
        else:
            return token

    def handle_define_sort(self, command: List):
        """Handle define-sort command."""
        _, sort_name, _, field_spec = command
        if isinstance(field_spec, list) and field_spec[0] == '_' and field_spec[1] == 'FiniteField':
            size = int(field_spec[2])
            if size <= 1:
                raise InvalidFieldSize(f"Invalid field size: {size}")
            self.ctx.field_size = size
            self.ctx.current_field = sort_name
            bits = (size - 1).bit_length()
            self.ctx.sorts[sort_name] = BitVecSort(bits)

    def evaluate_expr(self, expr, let_bindings=None) -> FieldExpr:
        """Evaluate expression to IR."""
        if let_bindings is None:
            let_bindings = {}

        if isinstance(expr, str):
            if expr in let_bindings:
                return let_bindings[expr]
            if expr in self.ctx.variables:
                return self.ctx.variables[expr]
            try:
                return FieldConst(int(expr))
            except ValueError:
                raise FieldError(f"Unknown identifier: {expr}")

        if not isinstance(expr, list):
            return expr

        op = expr[0]

        if op == 'let':
            bindings = expr[1]
            new_bindings = dict(let_bindings)
            for binding in bindings:
                var_name, value = binding
                new_bindings[var_name] = self.evaluate_expr(value, let_bindings)
            return self.evaluate_expr(expr[2], new_bindings)

        if op == 'ff.mul':
            args = [self.evaluate_expr(arg, let_bindings) for arg in expr[1:]]
            return FieldMul(args)

        if op == 'ff.add':
            args = [self.evaluate_expr(arg, let_bindings) for arg in expr[1:]]
            return FieldAdd(args)

        if op == 'and':
            args = [self.evaluate_expr(arg, let_bindings) for arg in expr[1:]]
            return FieldAnd(args)

        if op == 'not':
            arg = self.evaluate_expr(expr[1], let_bindings)
            return FieldNot(arg)

        if op == '=':
            left = self.evaluate_expr(expr[1], let_bindings)
            right = self.evaluate_expr(expr[2], let_bindings)
            return FieldEq(left, right)

        if op == '=>':
            left = self.evaluate_expr(expr[1], let_bindings)
            right = self.evaluate_expr(expr[2], let_bindings)
            return BoolImplies(left, right)

        if op == 'or':
            args = [self.evaluate_expr(arg, let_bindings) for arg in expr[1:]]
            return BoolOr(args)

        if op == 'ite':
            condition = self.evaluate_expr(expr[1], let_bindings)
            then_expr = self.evaluate_expr(expr[2], let_bindings)
            else_expr = self.evaluate_expr(expr[3], let_bindings)
            return FieldITE(condition, then_expr, else_expr)

        if op == 'as':
            if expr[1].startswith('ff'):
                value = int(expr[1][2:])
                return FieldConst(value)

        raise FieldError(f"Unsupported operation: {op}")

    def handle_assert(self, command: List):
        """Handle assert command."""
        ir_expr = self.evaluate_expr(command[1])
        z3_expr = ir_expr.to_z3(self.ctx)
        self.ctx.solver.add(z3_expr)

    def parse_commands(self, commands: List):
        """Parse and handle SMT-LIB commands."""
        for command in commands:
            if not isinstance(command, list):
                continue

            cmd_type = command[0]
            if cmd_type in ['set-logic', 'set-info']:
                continue
            elif cmd_type == 'define-sort':
                self.handle_define_sort(command)
            elif cmd_type == 'declare-fun':
                _, var_name, args, sort = command
                if args:  # We only support nullary functions
                    raise FieldError(f"Non-nullary functions not supported: {var_name}")

                # Handle different sorts
                if sort == 'Bool':
                    self.ctx.add_variable(var_name, 'Bool')
                elif sort in self.ctx.sorts:  # Field sort
                    self.ctx.add_variable(var_name, sort)
                else:
                    raise FieldError(f"Unknown sort: {sort}")
            elif cmd_type == 'assert':
                self.handle_assert(command)
            elif cmd_type == 'check-sat':
                logger.info("Solving constraints...")
                return self.ctx.solver.check()

    def parse_smt(self, smt_input: str):
        """Parse SMT-LIB input and solve."""
        tokens = self.tokenize(smt_input)
        commands = []
        while tokens:
            commands.append(self.parse_sexp(tokens))
        return self.parse_commands(commands)


def demo():
    """Demonstration of the enhanced parser."""
    smt_input = """
(set-info :smt-lib-version 2.6)
(set-info :category "crafted")
(set-info :status "unsat")
(set-logic QF_FFA)
(define-sort FF0 () (_ FiniteField 17))
(declare-fun a () Bool)
(declare-fun b () Bool)
(declare-fun return_n0 () FF0)
(declare-fun mul_n3 () FF0)
(declare-fun a_n2 () FF0)
(declare-fun mul_n4 () FF0)
(declare-fun b_n1 () FF0)
    """
    parser = EnhancedSMTParser()
    result = parser.parse_smt(smt_input)
    print(f"Result: {result}")
    if result == sat:
        print("Model:", parser.ctx.solver.model())


if __name__ == '__main__':
    demo()
