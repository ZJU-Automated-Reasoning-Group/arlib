"""
Inspect the formula (logic type, syntax features, etc)
  - Z3-based implementation
  - pySMT-based implementation (TODO)

In some "auto-config" mode, the user does not want to specify the type of a formula.
So, we may need to automatically infer the logic. Otherwise, we need to use
(set-logic ALL) for the theory solver, which may have poor performance.
"""

import z3


class FormulaInfo:
    def __init__(self, fml_str: str):
        self.formula = z3.And(z3.parse_smt2_string(fml_str))
        self.has_quantifier = self.has_quantifier()

    def apply_probe(self, name):
        g = z3.Goal()
        g.add(self.formula)
        p = z3.Probe(name)
        return p(g)

    def has_quantifier(self):
        return self.apply_probe('has-quantifiers')

    def get_logic(self):
        try:
            if self.apply_probe("is-propositional"):
                return "QF_UF"
            elif self.apply_probe("is-qfbv"):
                return "QF_BV"
            elif self.apply_probe("is-qfaufbv"):
                return "QF_AUFBV"
            elif self.apply_probe("is-qflia"):
                return "QF_LIA"
            elif self.apply_probe("is-qflra"):
                return "QF_LRA"
            elif self.apply_probe("is-qflira"):
                return "QF_LIRA"
            elif self.apply_probe("is-qfnia"):
                return "QF_NIA"
            elif self.apply_probe("is-qfnra"):
                return "QF_NRA"
            elif self.apply_probe("is-qfufnra"):
                return "QF_UFNRA"
            else:
                return "ALL"
        except Exception as ex:
            print(ex)
            return "ALL"
