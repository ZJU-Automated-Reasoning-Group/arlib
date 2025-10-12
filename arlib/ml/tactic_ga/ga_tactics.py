#!/usr/bin/env python3

"""
Genetic Algorithm for Optimizing Z3 Tactic Sequences

This module implements a genetic algorithm that evolves Z3 tactic sequences to find
optimal combinations of tactics for solving SMT problems efficiently. The algorithm
uses evolutionary principles to search through the space of possible tactic configurations.

Key Components:
- Tactic: Represents individual Z3 tactics with their parameters
- TacticSeq: Represents sequences of tactics that can be evolved
- GA: Genetic algorithm implementation with selection, crossover, and mutation

The system evaluates tactic sequences by running them on test problems and measuring
their performance in terms of solving time and success rate.

Author: Tactic Optimization Team
"""

import json
import os
import random
import subprocess
import time

import z3


class Param:
    """
    Represents a parameter for a Z3 tactic.

    Z3 tactics can have various types of parameters (boolean, integer, double, string, etc.).
    This class encapsulates a tactic parameter with its name, value, and type information.
    """
    def __init__(self, key, value, kind):
        """
        Initialize a tactic parameter.

        Args:
            key (str): Parameter name
            value: Parameter value (can be None for uninitialized)
            kind: Z3 parameter type (Z3_PK_BOOL, Z3_PK_UINT, etc.)
        """
        self.key = key
        self.value = value
        self.kind = kind

    def _replace(self, **kwargs):
        """
        Create a copy of this parameter with modified attributes.

        Args:
            **kwargs: Attributes to modify

        Returns:
            Param: New parameter instance with modified attributes
        """
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
    """
    Generate a random value for a tactic parameter based on its type.

    This function creates appropriate random values for different Z3 parameter types
    within reasonable bounds for effective tactic optimization.

    Args:
        param (Param): Parameter specification with type information
        rand (random.Random): Random number generator instance

    Returns:
        Random value appropriate for the parameter type

    Raises:
        LookupError: If parameter type is not recognized
    """
    if param.kind == z3.Z3_PK_BOOL:
        return True if rand.random() < 0.5 else False
    elif param.kind == z3.Z3_PK_UINT:
        return rand.randint(0, 4294967295)  # Full unsigned int range
    elif param.kind == z3.Z3_PK_DOUBLE:
        return rand.uniform(0.0, 1000000.0)  # Reasonable range for most tactic params
    elif param.kind == z3.Z3_PK_STRING:
        return ""  # Empty string for string parameters
    elif param.kind == z3.Z3_PK_SYMBOL:
        return ""  # Empty symbol for symbol parameters
    elif param.kind == z3.Z3_PK_OTHER:
        return None  # None for other/unknown parameter types
    elif param.kind == z3.Z3_PK_INVALID:
        return None  # None for invalid parameter types
    else:
        raise LookupError(f"Unknown parameter kind: {param.kind}")


class Tactic:
    """
    Represents a single Z3 tactic with its parameters.

    A tactic is a transformation or solving strategy in Z3. Each tactic can have
    multiple parameters that control its behavior. This class encapsulates a tactic
    and provides methods for mutation during the genetic algorithm evolution.
    """
    _mutate_probability = 0.02  # Probability of mutating each parameter
    _tactic_names = z3.tactics()  # Available Z3 tactic names
    _rand = random.Random()  # Random number generator for this class
    _rand.seed()  # Initialize with system time

    def __init__(self, name):
        """
        Initialize a tactic with its parameters.

        Args:
            name (str): Name of the Z3 tactic (e.g., "simplify", "solve-eqs")
        """
        self.name = name
        self.params = {}

        # Extract parameter descriptions from Z3 tactic
        pds = z3.Tactic(name).param_descrs()
        for i in range(pds.size()):
            pname = pds.get_name(i)
            self.params[pname] = Param(pname, None, pds.get_kind(pname))

    def clone(self):
        """
        Create a deep copy of this tactic.

        Returns:
            Tactic: Independent copy of this tactic
        """
        res = Tactic(self.name)
        res.params = self.params.copy()
        return res

    def as_json(self):
        """
        Serialize this tactic to JSON format.

        Returns:
            str: JSON representation of the tactic
        """
        return json.dumps(self, cls=CustomJsonEncoder, indent=2)

    def mutate(self):
        """
        Randomly mutate tactic parameters with low probability.

        Returns:
            Tactic: This tactic (modified in-place)
        """
        for key, param in self.params.items():
            if Tactic._rand.random() < Tactic._mutate_probability:
                self.params[key] = param._replace(value=random_param(param, Tactic._rand))
        return self

    @staticmethod
    def random():
        """
        Create a random tactic by selecting a random tactic name.

        Returns:
            Tactic: New tactic with random name
        """
        tactic_name = Tactic._rand.sample(Tactic._tactic_names, 1)[0]
        return Tactic(tactic_name)

    def __repr__(self):
        """String representation of the tactic for debugging."""
        return "{0} -> {1}".format(self.name, self.params)


class TacticSeq:
    """
    Represents a sequence of Z3 tactics that can be evolved using genetic algorithms.

    A tactic sequence is an ordered collection of tactics that are applied in sequence
    to solve SMT problems. The genetic algorithm evolves these sequences to find optimal
    combinations that solve problems efficiently.

    The default sequence includes common Z3 tactics that work well together for
    general SMT problem solving.
    """
    _max_size = 16  # Maximum number of tactics in a sequence

    _rand = random.Random()  # Random number generator for this class
    _rand.seed()  # Initialize with system time

    def __init__(self):
        """
        Initialize a tactic sequence with a default set of tactics.

        The default sequence includes tactics that are commonly effective for
        SMT problem solving, arranged in an order that typically works well.
        """
        self.storage = [Tactic(name) for name in [
            "simplify",          # Simplify formulas
            "propagate-values",  # Propagate constant values
            "solve-eqs",         # Solve equations
            "elim-uncnstr",      # Eliminate unconstrained variables
            "simplify",          # Simplify again after transformations
            "max-bv-sharing",    # Maximize bit-vector sharing
            "smt"                # Final SMT solving
        ]]
        self.fitness = 0.0  # Fitness score (lower time = higher fitness)

    def mutate(self):
        """
        Mutate all tactics in this sequence.

        Returns:
            TacticSeq: This sequence (modified in-place)
        """
        self.storage = list(map(lambda t: t.mutate(), self.storage))
        return self

    def to_z3_tactic(self):
        """
        Convert this tactic sequence to a Z3 Tactic object.

        This method creates a Z3 AndThen tactic that applies all tactics in the
        sequence in order. Tactics with parameters are created using With(),
        while simple tactics use Tactic() directly.

        Returns:
            z3.Tactic: Z3 tactic object representing this sequence

        Example:
            AndThen(With('simplify', param1=value1), Tactic('smt'))
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
    """
    Run test suite and measure execution time.

    This function runs the test suite using the 'run-tests' script with a timeout
    of 8 seconds. It measures the total execution time and returns it as the
    fitness score (lower time = better fitness).

    Returns:
        float: Execution time in seconds, or large penalty value if tests fail/timeout

    Note:
        - Success: Returns actual execution time
        - Failure/Timeout: Returns 4294967295 (large penalty value)
    """
    cmd = "timeout 8s ./run-tests *"
    start = time.time()
    ret = subprocess.call(cmd.split(), stdout=DEVNULL, stderr=DEVNULL)
    end = time.time()
    return end - start if 0 == ret else 4294967295


class GA:
    """
    Genetic Algorithm for evolving optimal Z3 tactic sequences.

    This class implements a complete genetic algorithm that evolves a population
    of tactic sequences over multiple generations. The algorithm uses tournament
    selection, crossover, and mutation to find tactic sequences that minimize
    test execution time.

    Population Management:
    - Population size: 64 individuals
    - Elite retention: Top 25% retained each generation
    - Selection: Tournament selection for breeding
    - Crossover: Single-point crossover of tactic sequences
    - Mutation: Parameter mutation with low probability
    """
    _population_size = 64      # Total population size
    _retain_percentage = 0.25  # Fraction of population to retain as elite
    _retain_size = int(_retain_percentage * _population_size) + 1

    _rand = random.Random()  # Random number generator for this class
    _rand.seed()  # Initialize with system time

    def __init__(self):
        """
        Initialize the genetic algorithm with a random population.

        Creates an initial population of mutated random tactic sequences
        that will be evolved over multiple generations.
        """
        self._population = [TacticSeq.random().mutate() for _ in range(self._population_size)]
        self._new = []       # New generation being evaluated
        self._retained = []  # Elite individuals retained from previous generation

    def evaluate(self):
        """
        Evaluate fitness of all individuals in the current population.

        For each tactic sequence in the population:
        1. Save it to z3.tactics file for Z3 to use
        2. Run the test suite and measure execution time
        3. Store the execution time as fitness (lower = better)
        """
        for tactics in self._population:
            tactics.dump("z3.tactics")
            tactics.fitness = run_tests()
            print(tactics.fitness)
            self._new.append(tactics)

    def dump(self):
        """
        Save the best tactic sequences from the current generation.

        Sorts the evaluated population by fitness and saves:
        1. Individual tactic files for each sequence
        2. Results summary with fitness scores
        """
        with open("z3.results", "w") as results:
            s = sorted(self._new, key=lambda e: e.fitness)
            for i, tactics in enumerate(s):
                tactics.dump(f"z3.tactics.{i}")
                results.write(f"{i}: {tactics.fitness}\n")

    def retained(self):
        """
        Save the elite individuals (best from previous generation).

        Saves the retained elite individuals separately for analysis
        and potential use in future generations.
        """
        with open("z3.results", "w") as results:
            for i, tactics in enumerate(self._retained):
                tactics.dump(f"z3.params.{i}")
                results.write(f"{i}: {tactics.fitness}\n")

    def repopulate(self):
        """
        Create next generation through selection, crossover, and mutation.

        This method implements the core genetic algorithm operators:
        1. Combine current and elite populations
        2. Select best individuals as elite for next generation
        3. Breed new individuals through crossover and mutation
        """
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
    """
    Run the complete genetic algorithm evolution process.

    This function runs the genetic algorithm for 128 generations:
    1. Evaluate current population fitness
    2. Save results and statistics
    3. Create next generation through breeding
    4. Repeat until convergence or generation limit

    The final elite population is saved for analysis.
    """
    ga = GA()
    for i in range(128):
        ga.evaluate()
        ga.dump()
        ga.repopulate()
    ga.retained()


if __name__ == "__main__":
    """
    Main execution: Demonstrate tactic sequence creation and visualization.

    When run directly, this script:
    1. Creates a random tactic sequence
    2. Displays it as a human-readable string
    3. Converts it to a Z3 tactic object
    4. Demonstrates the tactic on a simple test formula

    To run the full genetic algorithm instead, uncomment the main() call.
    """
    tactic_seq = TacticSeq.random()
    print("Randomly generated tactic sequence:")
    print(tactic_seq.to_string())
    print("\nConverting to Z3 tactic object...")
    z3_tactic = tactic_seq.to_z3_tactic()
    print("Demonstrating tactic on test formula...")
    pretty_print_tactic(z3_tactic)

    # Uncomment the following line to run the full genetic algorithm
    # main()
