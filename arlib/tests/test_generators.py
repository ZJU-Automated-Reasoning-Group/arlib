# coding: utf-8
import z3
from arlib.tests import TestCase, main
from arlib.tests.grammar_gene import generate_from_grammar_as_str


class TestGenerator(TestCase):

    def test_grammar_gene(self):
        try:
            fmlstr = generate_from_grammar_as_str(logic="QF_BV")
            if not fmlstr:  # generation error?
                self.skipTest("Formula generation error")
            fml = z3.And(z3.parse_smt2_string(fmlstr))
            # Just verify we can parse the formula
            self.assertIsNotNone(fml)
        except Exception as ex:
            self.fail(f"Exception occurred: {str(ex)}")


if __name__ == '__main__':
    main()
