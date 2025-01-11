"""
Translated from https://github.com/pysmt/pysmt/blob/master/examples/model_checking.py by LLM

# It provides a simple implementation of Bounded Model Checking [1]
# with K-Induction [2], and PDR [3,4], and applies it on a simple
# transition system.
#
# [1] Biere, Cimatti, Clarke, Zhu,
#     "Symbolic Model Checking without BDDs",
#     TACAS 1999
#
# [2] Sheeran, Singh,  Stalmarck,
#     "Checking  safety  properties  using  induction  and  a SAT-solver",
#     FMCAD 2000
#
# [3] Bradley
#     "SAT-Based Model Checking without Unrolling",
#     VMCAI 2011
#
# [4] Een, Mischenko, Brayton
#     "Efficient implementation of property directed reachability",
#     FMCAD 2011

"""
from z3 import *


def next_var(v):
    """Returns the 'next' of the given variable"""
    if is_bool(v):
        return Bool(f"next({v})")
    else:
        return BitVec(f"next({v})", v.size())


def at_time(v, t):
    """Builds an SMT variable representing v at time t"""
    if is_bool(v):
        return Bool(f"{v}@{t}")
    else:
        return BitVec(f"{v}@{t}", v.size())


class TransitionSystem(object):
    """Trivial representation of a Transition System."""

    def __init__(self, variables, init, trans):
        self.variables = variables
        self.init = init
        self.trans = trans


class PDR(object):
    def __init__(self, system):
        self.system = system
        self.frames = [system.init]
        self.solver = Solver()
        self.prime_map = {v: next_var(v) for v in self.system.variables}

    def check_property(self, prop):
        print(f"Checking property {prop}...")

        while True:
            cube = self.get_bad_state(prop)
            if cube is not None:
                if self.recursive_block(cube):
                    print(f"--> Bug found at step {len(self.frames)}")
                    break
                else:
                    print(f"   [PDR] Cube blocked '{cube}'")
            else:
                if self.inductive():
                    print("--> The system is safe!")
                    break
                else:
                    print(f"   [PDR] Adding frame {len(self.frames)}...")
                    self.frames.append(True)

    def get_bad_state(self, prop):
        return self.solve(And(self.frames[-1], Not(prop)))

    def solve(self, formula):
        self.solver.push()
        self.solver.add(formula)
        if self.solver.check() == sat:
            model = self.solver.model()
            result = And([v == model.eval(v) for v in self.system.variables])
            self.solver.pop()
            return result
        self.solver.pop()
        return None

    def recursive_block(self, cube):
        for i in range(len(self.frames) - 1, 0, -1):
            cubeprime = substitute(cube,
                                   [(v, next_var(v)) for v in self.system.variables])
            cubepre = self.solve(And(self.frames[i - 1],
                                     self.system.trans,
                                     Not(cube),
                                     cubeprime))
            if cubepre is None:
                for j in range(1, i + 1):
                    self.frames[j] = And(self.frames[j], Not(cube))
                return False
            cube = cubepre
        return True

    def inductive(self):
        if len(self.frames) > 1:
            s = Solver()
            s.add(Not(self.frames[-1] == self.frames[-2]))
            return s.check() == unsat
        return False


class BMCInduction(object):
    def __init__(self, system):
        self.system = system

    def get_subs(self, i):
        subs_i = {}
        for v in self.system.variables:
            subs_i[v] = at_time(v, i)
            subs_i[next_var(v)] = at_time(v, i + 1)
        return subs_i

    def get_unrolling(self, k):
        res = []
        for i in range(k + 1):
            subs_i = self.get_subs(i)
            res.append(substitute(self.system.trans,
                                  [(v, subs_i[v]) for v in subs_i]))
        return And(res)

    def get_simple_path(self, k):
        res = []
        for i in range(k + 1):
            subs_i = self.get_subs(i)
            for j in range(i + 1, k + 1):
                state = []
                subs_j = self.get_subs(j)
                for v in self.system.variables:
                    v_i = substitute(v, [(v, subs_i[v])])
                    v_j = substitute(v, [(v, subs_j[v])])
                    state.append(v_i != v_j)
                res.append(Or(state))
        return And(res)

    def get_k_hypothesis(self, prop, k):
        res = []
        for i in range(k):
            subs_i = self.get_subs(i)
            res.append(substitute(prop, [(v, subs_i[v])
                                         for v in self.system.variables]))
        return And(res)

    def get_bmc(self, prop, k):
        init_0 = substitute(self.system.init,
                            [(v, self.get_subs(0)[v])
                             for v in self.system.variables])
        prop_k = substitute(prop,
                            [(v, self.get_subs(k)[v])
                             for v in self.system.variables])
        return And(self.get_unrolling(k), init_0, Not(prop_k))

    def get_k_induction(self, prop, k):
        subs_k = self.get_subs(k)
        prop_k = substitute(prop,
                            [(v, subs_k[v]) for v in self.system.variables])
        return And(self.get_unrolling(k),
                   self.get_k_hypothesis(prop, k),
                   self.get_simple_path(k),
                   Not(prop_k))

    def check_property(self, prop):
        print(f"Checking property {prop}...")
        for b in range(100):
            f = self.get_bmc(prop, b)
            print(f"   [BMC]    Checking bound {b + 1}...")
            s = Solver()
            s.add(f)
            if s.check() == sat:
                print(f"--> Bug found at step {b + 1}")
                return

            f = self.get_k_induction(prop, b)
            print(f"   [K-IND]  Checking bound {b + 1}...")
            s = Solver()
            s.add(f)
            if s.check() == unsat:
                print("--> The system is safe!")
                return


def counter(bit_count):
    bits = BitVec("bits", bit_count)
    nbits = next_var(bits)
    reset = Bool("r")
    nreset = next_var(reset)
    variables = [bits, reset]

    init = And(bits == 0, Not(reset))

    trans = And(nbits == bits + 1,
                (nbits == 0) == nreset)

    true_prop = Implies(reset, bits == 0)
    false_prop = bits != 2 ** bit_count - 1

    return (TransitionSystem(variables, init, trans), [true_prop, false_prop])


def main():
    example = counter(4)
    bmcind = BMCInduction(example[0])
    pdr = PDR(example[0])

    for prop in example[1]:
        bmcind.check_property(prop)
        pdr.check_property(prop)
        print("")


if __name__ == "__main__":
    main()