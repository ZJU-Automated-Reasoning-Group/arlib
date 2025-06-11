#!/usr/bin/env python3
"""
Backbone Example - Computing Backbone Literals

This example demonstrates arlib's backbone computation capabilities.
Backbone literals are literals that are true in all satisfying assignments.
"""

import z3
from arlib.backbone.smt_backbone_literals import get_backbone_literals


def basic_backbone_example():
    """Demonstrate basic backbone computation."""
    print("=== Basic Backbone ===")
    
    a, b, c = z3.Bools('a b c')
    formula = z3.And(z3.Or(a, b), z3.Implies(a, c))
    literals = [a, z3.Not(a), b, z3.Not(b), c, z3.Not(c)]
    
    try:
        backbone = get_backbone_literals(formula, literals, 'model_enumeration')
        print(f"Backbone literals: {len(backbone)} found")
        for lit in backbone:
            print(f"  {lit}")
    except Exception as e:
        print("Backbone computation failed")


def algorithm_comparison():
    """Compare different backbone algorithms."""
    print("\n=== Algorithm Comparison ===")
    
    x, y = z3.Ints('x y')
    formula = z3.And(x + y >= 5, x >= 2, y >= 1, x <= 10, y <= 10)
    literals = [x >= 3, x <= 8, y >= 2, y <= 7]
    
    algorithms = ['model_enumeration', 'sequence_checking']
    
    for alg in algorithms:
        try:
            backbone = get_backbone_literals(formula, literals, alg)
            print(f"{alg}: {len(backbone)} backbone literals")
        except Exception as e:
            print(f"{alg}: failed")


def main():
    """Run all backbone examples."""
    print("Backbone Examples - arlib's Backbone Analysis")
    print("=" * 45)
    
    basic_backbone_example()
    algorithm_comparison()
    
    print("\nBackbone examples completed!")


if __name__ == "__main__":
    main() 