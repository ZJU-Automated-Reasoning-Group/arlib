import pycvc5
import random
import time

DEFAULT_FILE = 'cvc.sl'

SYGUS_SOLVERS_TO_FILE = True
NON_SYGUS_TO_FILE = False

def define_fun_to_string(f, params, body):
    sort = f.getSort()
    if sort.isFunction():
        sort = f.getSort().getFunctionCodomainSort()
    result = ""
    result += "(define-fun " + str(f) + " ("
    for i in range(0, len(params)):
        if i > 0:
            result += " "
        else:
            result += "(" + str(params[i]) + " " + str(params[i].getSort()) + ")"
    result += ") " + str(sort) + " " + str(body) + ")"
    return result


def print_synth_solutions(terms, sols):
    result = ""
    for i in range(0, len(terms)):
        params = []
        if sols[i].getKind() == pycvc5.kinds.Lambda:
            params += sols[i][0]
            body = sols[i][1]
        result += "  " + define_fun_to_string(terms[i], params, body) + "\n"
    print(result)

# Associates with each CVC Solver a dictionary of python object to CVC Term conversions
# This includes converting python variables to CVC constants and converting python classes to Sort
# This should only be used by to_cvc to cache Terms that have already been declared in a given Solver
DEFINITIONS = dict()

# For __str__ of CVCObject, stores the infix for each operator for printing
# noinspection PyUnresolvedReferences
INFIXES = {pycvc5.kinds.And: ' & ',
           pycvc5.kinds.Or: ' | ',
           pycvc5.kinds.Implies: ' >> ',
           pycvc5.kinds.Equal: ' == ',
           pycvc5.kinds.Distinct: ' != ',
           pycvc5.kinds.Geq: ' >= ',
           pycvc5.kinds.Leq: ' <= ',
           pycvc5.kinds.Gt: ' > ',
           pycvc5.kinds.Lt: ' < ',
           pycvc5.kinds.Plus: ' + '}

PREFIXES = {pycvc5.kinds.And: 'and',
           pycvc5.kinds.Or: 'or',
           pycvc5.kinds.Implies: '=>',
           pycvc5.kinds.Equal: '=',
           pycvc5.kinds.Distinct: 'distinct',
           pycvc5.kinds.Geq: '>=',
           pycvc5.kinds.Leq: '<=',
           pycvc5.kinds.Gt: '>',
           pycvc5.kinds.Lt: '<',
           pycvc5.kinds.Not: 'not',
           pycvc5.kinds.Plus: '+'}


SYGUS_SOLVERS = set()
SYNTHESIS_FUNCTIONS = dict()
FILE_POINTERS = dict()

def cvc_repr(i):
    return repr(i).lower() if isinstance(i, bool) else repr(i)


# Represents a CVC Term in a Solver-agnostic way.
# noinspection PyUnresolvedReferences
class CVCObject:
    # if this CVCObject should be constructed with mkTerm, kind represents the CVC Kind
    # and the operands are a tuple of CVCObjects (or any object with to_cvc support)
    def __init__(self, kind=None, operands: tuple = None, name=None):
        self.kind = kind
        self.operands = operands
        self.name = name

        # Such that getting self.hash does not trigger __getattr__
        self.hash = None

    # dot-notation access of a variable.
    def __getattr__(self, key):
        return CVCObject(kind=pycvc5.kinds.ApplySelector, operands=(key, self))

    def __and__(self, rhs):
        #if rhs is True:
        #    return self
        #if rhs is False:
        #    return False
        return CVCObject(kind=pycvc5.kinds.And, operands=(self, rhs))

    def __or__(self, rhs):
        #if rhs is False:
        #    return self
        #if rhs is True:
        #    return True
        return CVCObject(kind=pycvc5.kinds.Or, operands=(self, rhs))

    def __eq__(self, other):
        return CVCObject(kind=pycvc5.kinds.Equal, operands=(self, other))

    def __ne__(self, other):
        return CVCObject(kind=pycvc5.kinds.Distinct, operands=(self, other))

    def __lt__(self, other):
        return CVCObject(kind=pycvc5.kinds.Lt, operands=(self, other))

    def __gt__(self, other):
        return CVCObject(kind=pycvc5.kinds.Gt, operands=(self, other))

    def __le__(self, other):
        return CVCObject(kind=pycvc5.kinds.Le, operands=(self, other))

    def __ge__(self, other):
        return CVCObject(kind=pycvc5.kinds.Geq, operands=(self, other))

    # python "not" cannot be overwritten
    def __invert__(self):
        return CVCObject(kind=pycvc5.kinds.Not, operands=(self,))

    def __rand__(self, lhs):
        #if lhs is True:
        #    return self
        #if lhs is False:
        #    return False
        return CVCObject(kind=pycvc5.kinds.And, operands=(lhs, self))

    def __ror__(self, lhs):
        #if lhs is False:
        #    return self
        #if lhs is True:
        #    return True
        return CVCObject(kind=pycvc5.kinds.Or, operands=(lhs, self))

    def __rshift__(self, rhs):
        #if rhs is True:
        #    return True
        return CVCObject(kind=pycvc5.kinds.Implies, operands=(self, rhs))

    def __rrshift__(self, lhs):
        #if lhs is False:
        #    return True
        #if lhs is True:
        #    return self
        return CVCObject(kind=pycvc5.kinds.Implies, operands=(lhs, self))

    def __repr__(self):
        if getattr(self, 'kind', None) is None:
            if self.name is None:
                return self.__class__.__name__ + str(id(self))
            else:
                return self.name
        elif self.kind == pycvc5.kinds.ApplySelector:
            return f"({self.operands[0]} {repr(self.operands[1])})"
        elif self.kind == pycvc5.kinds.ApplyConstructor:
            return self.operands[1]
        elif self.kind == pycvc5.kinds.ApplyUf:
            return f"({self.operands[0].__name__} {' '.join(map(cvc_repr, self.operands[1:]))})"
        return f"({PREFIXES[self.kind]} {' '.join(map(cvc_repr, self.operands))})"

    def __str__(self):
        if getattr(self, 'kind', None) is None:
            if self.name is None:
                return self.__class__.__name__ + str(id(self))[-3:]
            else:
                return self.name
        elif self.kind == pycvc5.kinds.ApplySelector:
            return f"{self.operands[1]}." + self.operands[0]
        elif self.kind == pycvc5.kinds.ApplyConstructor:
            return self.operands[0].__name__ + '.' + self.operands[1]
        elif self.kind == pycvc5.kinds.Not:
            return f"~{str(self.operands[0])}"
        return f"({INFIXES[self.kind].join(map(str, self.operands))})"

    # hash is identical among CVCObjects with identical contents
    def __hash__(self):
        if getattr(self, 'hash', None) is not None:
            return self.hash
        if getattr(self, 'kind', None) is None:
            return id(self)
        return hash((self.kind, tuple(self.operands)))

    # TODO setattr

    def __add__(self, rhs):
        return CVCObject(kind=pycvc5.kinds.Plus, operands=(self, rhs))

    def __radd__(self, lhs):
        return CVCObject(kind=pycvc5.kinds.Plus, operands=(lhs, self))


# handles accessing Enum variables
# noinspection PyUnresolvedReferences
class CVCEnumMetaclass(type):
    def __getattr__(cls, key):
        result = CVCObject()
        result.kind = pycvc5.kinds.ApplyConstructor
        result.operands = [cls, key]
        return result


class CVCEnum(CVCObject, metaclass=CVCEnumMetaclass):
    pass


class CVCInt(CVCObject):
    pass


class CVCBool(CVCObject):
    def __init__(self, value=None, name=None):
        self.value = value
        super().__init__(name=name)

    def __hash__(self):
        if self.value is not None:
            return hash(self.value)
        return id(self)

    def __repr__(self):
        if self.value is not None:
            return 'true' if self.value else 'false'
        return super().__repr__()

    def __str__(self):
        if self.value is not None:
            return str(self.value)
        return super().__str__()


class CVCVar(CVCObject):
    def __init__(self, var):
        self.var = var
        super().__init__()


def file_pointer(solver):
    if solver not in FILE_POINTERS:
        FILE_POINTERS[solver] = open(DEFAULT_FILE, 'w')
    return FILE_POINTERS[solver]


def set_logic(solver):
    solver.setLogic("ALL")
    if solver in SYGUS_SOLVERS:
        solver.setOption('incremental', 'false')
        solver.setOption("lang", "sygus2")
    if solver in SYGUS_SOLVERS and SYGUS_SOLVERS_TO_FILE:
        file_pointer(solver).write('(set-logic ALL)\n')


def check(solver, quiet=False):
    if NON_SYGUS_TO_FILE or (solver in SYGUS_SOLVERS and SYGUS_SOLVERS_TO_FILE):
        file_pointer(solver).write('(check-synth)\n' if solver in SYGUS_SOLVERS else '(check-sat)\n')
        # start = time.time()
        # os.system(f'cvc5 temp/{id(solver)}.sl')
        # print(time.time() - start)
        file_pointer(solver).flush()
    else:
        start = time.time()
        check = solver.checkSynth() if solver in SYGUS_SOLVERS else solver.checkSat()
        if not quiet:
            print(check)
        if check.isUnsat() and solver in SYGUS_SOLVERS:
            print_synth_solutions(SYNTHESIS_FUNCTIONS[solver],
                                  solver.getSynthSolutions(SYNTHESIS_FUNCTIONS[solver]))
        if not quiet:
            print(time.time() - start)
        return check.isSat()



# noinspection PyUnresolvedReferences
def to_cvc(term, solver):
    # things we do not expect to be in the DEFINITIONS dictionary
    if isinstance(term, pycvc5.Term) or isinstance(term, pycvc5.Sort) or isinstance(term, pycvc5.Datatype):
        return term
    if isinstance(term, CVCBool) and hasattr(term, 'value'):
        return solver.mkBoolean(term.value)
    if isinstance(term, bool):
        return solver.mkBoolean(term)
    if isinstance(term, int):
        return solver.mkInteger(term)

    # Add the solver to DEFINITIONS if not present
    if solver not in DEFINITIONS:
        DEFINITIONS[solver] = {CVCInt: solver.getIntegerSort(), CVCBool: solver.getBooleanSort(),
                               int: solver.getIntegerSort(), bool: solver.getBooleanSort()}
    # return the cached result if cached
    if term in DEFINITIONS[solver]:
        return DEFINITIONS[solver][term]

    if isinstance(term, type):
        # declare Enum or Record
        if issubclass(term, CVCEnum):
            sort = solver.mkDatatypeDecl(term.__name__)
            tuples = ' '.join([f"({k})" for (k, _) in term.__annotations__.items()])
            if NON_SYGUS_TO_FILE or solver in SYGUS_SOLVERS:
                file_pointer(solver).write(f"(declare-datatype {term.__name__} ({tuples}))\n")
            for (k, _) in term.__annotations__.items():
                sort.addConstructor(solver.mkDatatypeConstructorDecl(k))
            DEFINITIONS[solver][term] = solver.mkDatatypeSort(sort)
        else:
            datatype = solver.mkDatatypeDecl(term.__name__)
            constructor = solver.mkDatatypeConstructorDecl("constructor")
            for (k, v) in term.__annotations__.items():
                constructor.addSelector(k, to_cvc(v, solver))
            datatype.addConstructor(constructor)
            tuples = ' '.join([f"({k} {to_cvc(v, solver)})" for (k, v) in term.__annotations__.items()])
            if NON_SYGUS_TO_FILE or solver in SYGUS_SOLVERS:
                file_pointer(solver).write(f"(declare-datatype {term.__name__} ((c{random.randint(0,10000)} {tuples})))\n")
            DEFINITIONS[solver][term] = solver.mkDatatypeSort(datatype) #solver.mkRecordSort([(k, to_cvc(v, solver)) for (k, v) in term.__annotations__.items()])
    elif callable(term) and term.sygus:
        # sygus function
        annotations = term.__annotations__
        return_type = annotations.pop('return')

        variables = []
        objects = []
        for (name, a) in annotations.items():
            var = solver.mkVar(to_cvc(a, solver))
            variables.append(var)
            obj = CVCObject(name=name)
            objects.append(obj)
            DEFINITIONS[solver][obj] = var
        inner = term(*objects)

        non_terminals = []
        rules_objects = []
        for (name, a) in inner.__annotations__.items():
            non_terminal = solver.mkVar(to_cvc(a, solver), name)
            non_terminals.append(non_terminal)
            obj = CVCObject(name=name)
            rules_objects.append(obj)
            DEFINITIONS[solver][obj] = non_terminal
        rules = inner(*rules_objects)

        grammar = solver.mkSygusGrammar(variables, non_terminals)
        for (k, v) in rules.items():
            grammar.addRules(to_cvc(k, solver), [to_cvc(nt, solver) for nt in v])
            #for nt in v:
            #    print(f"(define-fun f{str(id(nt))[-4:]} () {to_cvc(k, solver).getSort()} {to_cvc(nt, solver)} )")
        DEFINITIONS[solver][term] = solver.synthFun(term.__name__, variables, to_cvc(return_type, solver), grammar)

        if solver not in SYNTHESIS_FUNCTIONS:
            SYNTHESIS_FUNCTIONS[solver] = []

        inputs = ' '.join([f"({k} {to_cvc(v, solver)})" for (k, v) in annotations.items()])
        nts = ' '.join([f"({k} {to_cvc(v, solver)})" for (k, v) in inner.__annotations__.items()])
        r = ' '.join([f"({a} {to_cvc(b, solver)} ({' '.join([f'{cvc_repr(nt)}' for nt in v])}))" for ((a, b), v) in zip(inner.__annotations__.items(), rules.values())])
        if solver in SYGUS_SOLVERS:
            file_pointer(solver).write(f"(synth-fun {term.__name__} ({inputs}) {to_cvc(return_type, solver)} ({nts}) ({r}) )\n")
        SYNTHESIS_FUNCTIONS[solver].append(DEFINITIONS[solver][term])
        annotations['return'] = return_type
    elif getattr(term, 'kind', None) is None:
        # declare constant
        sort = to_cvc(term.__class__, solver)
        if solver in SYGUS_SOLVERS:
            if SYGUS_SOLVERS_TO_FILE:
                file_pointer(solver).write(f'(declare-var {cvc_repr(term)} {to_cvc(term.__class__, solver)})\n')
            DEFINITIONS[solver][term] = solver.mkSygusVar(sort, str(term))
        else:
            if NON_SYGUS_TO_FILE:
                file_pointer(solver).write(f'(declare-const {cvc_repr(term)} {to_cvc(term.__class__, solver)})\n')
            DEFINITIONS[solver][term] = solver.mkConst(sort, str(term))
    elif term.kind == pycvc5.kinds.ApplySelector:
        obj = to_cvc(term.operands[1], solver)
        DEFINITIONS[solver][term] = solver.mkTerm(pycvc5.kinds.ApplySelector,
                                                  obj.getSort().getDatatype()[0].getSelectorTerm(term.operands[0]), obj)
    elif term.kind == pycvc5.kinds.ApplyConstructor:
        sort = to_cvc(term.operands[0], solver)
        DEFINITIONS[solver][term] = solver.mkTerm(pycvc5.kinds.ApplyConstructor,
                                                  sort.getDatatype().getConstructorTerm(term.operands[1]))
    elif term.kind == pycvc5.kinds.ApplyUf and not term.operands[0].sygus and term.operands[0] not in DEFINITIONS[solver]:
        annotations = term.operands[0].__annotations__
        variables = []
        objects = []
        for o, name in zip(term.operands[1:], term.operands[0].varnames):
            var = solver.mkVar(to_cvc(o, solver).getSort())
            variables.append(var)
            obj = CVCObject(name=name)
            objects.append(obj)
            DEFINITIONS[solver][obj] = var
        out = term.operands[0](*objects)
        output = to_cvc(out, solver)

        DEFINITIONS[solver][term.operands[0]] = solver.defineFun(term.operands[0].__name__, variables, output.getSort(), output)
        DEFINITIONS[solver][term] = solver.mkTerm(term.kind, *map(lambda x: to_cvc(x, solver), term.operands))

        inputs = ' '.join([f"({name} {to_cvc(o, solver).getSort()})" for (o, name) in zip(term.operands[1:], term.operands[0].varnames)])
        if NON_SYGUS_TO_FILE or solver in SYGUS_SOLVERS:
            file_pointer(solver).write(
                f"(define-fun {term.operands[0].__name__} ({inputs}) {output.getSort()} {cvc_repr(out)})\n")
    else:
        # remaining kinds use mkTerm
        """m = list(map(lambda x: to_cvc(x, solver), term.operands))
        try:
            DEFINITIONS[solver][term] = solver.mkTerm(term.kind, *m)
        except:
            print(term)"""
        DEFINITIONS[solver][term] = solver.mkTerm(term.kind, *map(lambda x: to_cvc(x, solver), term.operands))
    # TODO handle synthesis functions and whether to make sygusVar vs Var.
    return DEFINITIONS[solver][term]


def assert_formula(solver, formula):
    if solver in SYGUS_SOLVERS:
        solver.addSygusConstraint(to_cvc(~formula, solver))
        if SYGUS_SOLVERS_TO_FILE:
            file_pointer(solver).write(f"(constraint {cvc_repr(~formula)})\n")
    else:
        solver.assertFormula(to_cvc(formula, solver))
        if NON_SYGUS_TO_FILE:
            file_pointer(solver).write(f"(assert {cvc_repr(formula)})\n")


def synthesize(f, name=None):
    # noinspection PyUnresolvedReferences
    def custom(*args):
        return f(*args)

    custom.__annotations__ = f.__annotations__
    custom.__name__ = f.__name__ if name is None else name
    custom.sygus = True

    def replaced(*args):
        return CVCObject(kind=pycvc5.kinds.ApplyUf, operands=(custom,) + args)
    return replaced


def define_fun(f, name=None):
    # send bound variables for each argument
    # simplify the term and create a function
    # replace f with a function application

    def custom(*args):
        return f(*args)

    custom.__annotations__ = f.__annotations__
    custom.__name__ = f.__name__ if name is None else name
    custom.sygus = False
    custom.varnames = f.__code__.co_varnames

    def g(*args):
        return CVCObject(kind=pycvc5.kinds.ApplyUf, operands=(custom,) + args)
    return g