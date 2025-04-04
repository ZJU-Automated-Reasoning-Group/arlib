#!/usr/bin/env python3

import json
import os
import random
import subprocess
import time

import z3


class Param:
    def __init__(self, key, value, kind):
        self.key = key
        self.value = value
        self.kind = kind

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Param):
            return {
                "key": o.key,
                "value": o.value
            }
        elif isinstance(o, Tactic):
            return {
                "name": o.name,
                "type": o.name,
                "params": filter(lambda p: p.value is not None, o.params.values())
            }
        return json.JSONEncoder.default(o)


def random_param(param, rand):
    if param.kind == z3.Z3_PK_BOOL:
        return True if rand.random() < 0.5 else False
    elif param.kind == z3.Z3_PK_UINT:
        return rand.randint(0, 4294967295)
    elif param.kind == z3.Z3_PK_DOUBLE:
        return rand.uniform(0.0, 1000000.0)
    elif param.kind == z3.Z3_PK_STRING:
        return ""
    elif param.kind == z3.Z3_PK_SYMBOL:
        return ""
    elif param.kind == z3.Z3_PK_OTHER:
        return None
    elif param.kind == z3.Z3_PK_INVALID:
        return None
    else:
        raise LookupError("Wat?")


class Tactic:
    _mutate_probability = 0.02
    _tactic_names = z3.tactics()
    _rand = random.Random()
    _rand.seed()

    def __init__(self, name):
        self.name = name
        self.params = {}

        pds = z3.Tactic(name).param_descrs()
        for i in range(pds.size()):
            pname = pds.get_name(i)
            self.params[pname] = Param(pname, None, pds.get_kind(pname))

    def clone(self):
        res = Tactic(self.name)
        res.params = self.params.copy()
        return res

    def as_json(self):
        return json.dumps(self, cls=CustomJsonEncoder, indent=2)

    def mutate(self):
        for key, param in self.params.items():
            if Tactic._rand.random() < Tactic._mutate_probability:
                self.params[key] = param._replace(value=random_param(param, Tactic._rand))
        return self

    @staticmethod
    def random():
        tactic_name = Tactic._rand.sample(Tactic._tactic_names, 1)[0]
        return Tactic(tactic_name)

    def __repr__(self):
        return "{0} -> {1}".format(self.name, self.params)


class TacticSeq:
    _max_size = 16

    _rand = random.Random()
    _rand.seed()

    def __init__(self):
        self.storage = [Tactic(name) for name in [
            "simplify",
            "propagate-values",
            "solve-eqs",
            "elim-uncnstr",
            "simplify",
            "max-bv-sharing",
            "smt"
        ]]
        self.fitness = 0.0

    def mutate(self):
        self.storage = list(map(lambda t: t.mutate(), self.storage))
        return self

    def to_z3_tactic(self):
        """
        Build a Tactic object from self.storage, such as the one below
        t2 = AndThen(With('simplify', blast_distinct=False, elim_and=True, flat=False, hoist_mul=True, local_ctx=False,
                  pull_cheap_ite=True, push_ite_bv=False, som=False),
             Tactic('elim-uncnstr'),
             Tactic('purify-arith'),
             Tactic('smt'))
        """
        if not self.storage:
            return z3.Tactic('skip')
            
        result = None
        for tactic in self.storage:
            # Create a base tactic with the name
            if tactic.params and any(p.value is not None for p in tactic.params.values()):
                # Handle tactic with parameters using With
                param_dict = {p.key: p.value for p in tactic.params.values() if p.value is not None}
                current = z3.With(tactic.name, **param_dict)
            else:
                # Simple tactic without parameters
                current = z3.Tactic(tactic.name)
            
            # Chain the tactics using AndThen
            if result is None:
                result = current
            else:
                result = z3.AndThen(result, current)
        
        return result

    def dump(self, fname):
        with open(fname, "w") as ffile:
            json.dump(self.storage, ffile, cls=CustomJsonEncoder, indent=2)
        return self

    @staticmethod
    def crossover(a, b):
        res = TacticSeq()
        crossover_point = TacticSeq._rand.randint(0, len(a.storage) - 1)
        res.storage = []
        res.storage.extend(a.storage[:crossover_point])
        res.storage.extend(b.storage[crossover_point:])
        res.storage = list(map(lambda t: t.clone(), res.storage))
        return res

    @staticmethod
    def random():
        res = TacticSeq()
        return res

    def to_string(self):
        """Return a string representation of the tactic sequence"""
        result = []
        for tactic in self.storage:
            params_str = ", ".join(f"{p.key}={p.value}" for p in tactic.params.values() if p.value is not None)
            if params_str:
                result.append(f"{tactic.name}({params_str})")
            else:
                result.append(tactic.name)
        return " -> ".join(result)


DEVNULL = open(os.devnull, "w")


def run_tests():
    cmd = "timeout 8s ./run-tests *"
    start = time.time()
    ret = subprocess.call(cmd.split(), stdout=DEVNULL, stderr=DEVNULL)
    end = time.time()
    return end - start if 0 == ret else 4294967295


class GA:
    _population_size = 64
    _retain_percentage = 0.25
    _retain_size = int(_retain_percentage * _population_size) + 1

    _rand = random.Random()
    _rand.seed()

    def __init__(self):
        self._population = [TacticSeq.random().mutate() for _ in range(self._population_size)]
        self._new = []
        self._retained = []

    def evaluate(self):
        for tactics in self._population:
            tactics.dump("z3.tactics")
            tactics.fitness = run_tests()
            print(tactics.fitness)
            self._new.append(tactics)

    def dump(self):
        with open("z3.results", "w") as results:
            s = sorted(self._new, key=lambda e: e.fitness)
            for i, tactics in enumerate(s):
                tactics.dump("z3.tactics.{0}".format(i))
                results.write("{0}: {1}\n".format(i, tactics.fitness))

    def retained(self):
        with open("z3.results", "w") as results:
            for i, tactics in enumerate(self._retained):
                tactics.dump("z3.params.{0}".format(i))
                results.write("{0}: {1}\n".format(i, tactics.fitness))

    def repopulate(self):
        self._new.extend(self._retained)
        self._new = sorted(self._new, key=lambda e: e.fitness)

        self._population = []
        self._retained = []

        self._retained.extend(self._new[:self._retain_size])

        while len(self._population) < self._population_size:
            i1 = self._rand.randint(0, len(self._new) - 1)
            i2 = self._rand.randint(0, len(self._new) - 1)
            if i1 == i2:
                continue
            elif self._new[i1].fitness == 4294967295 or self._new[i2].fitness == 4294967295:
                continue
            else:
                self._population.append(
                    TacticSeq.crossover(
                        self._new[i1],
                        self._new[i2]
                    ).mutate()
                )

        self._new = []


def pretty_print_tactic(tactic):
    """
    Print Z3 tactic in a readable format by applying it to a simple formula
    and showing its string representation
    """
    x = z3.Int('x')
    y = z3.Int('y')
    formula = z3.And(x > 0, y > 0, x + y == 10)
    goal = z3.Goal()
    goal.add(formula)
    result = tactic(goal)
    
    print("Z3 tactic applied to test formula (x > 0 ∧ y > 0 ∧ x + y == 10):")
    for i, subgoal in enumerate(result):
        print(f"Subgoal {i+1}:")
        for j, formula in enumerate(subgoal):
            print(f"  {j+1}. {formula}")


def main():
    ga = GA()
    for i in range(128):
        ga.evaluate()
        ga.dump()
        ga.repopulate()
    ga.retained()


if __name__ == "__main__":
    tactic_seq = TacticSeq.random()
    print("Tactic sequence as string:")
    print(tactic_seq.to_string())
    print("\nTactic sequence as Z3 tactic object:")
    z3_tactic = tactic_seq.to_z3_tactic()
    pretty_print_tactic(z3_tactic)
    # main()
