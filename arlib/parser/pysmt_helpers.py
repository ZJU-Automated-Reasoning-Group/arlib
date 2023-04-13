# from https://github.com/neta-elad/qifac

import io
from typing import Any
from typing import Any, Dict, Optional, Set, TextIO, Tuple

from pysmt.operators import ALL_TYPES, FORALL
from pysmt.smtlib.parser import SmtLibParser
from pysmt.smtlib.script import SmtLibScript
from pysmt.walkers import TreeWalker, handles

Term = Any

all_types_but_forall = list(ALL_TYPES)
all_types_but_forall.remove(FORALL)


class AbstractForallWalker(TreeWalker):
    def __init__(self) -> None:
        TreeWalker.__init__(self)

    @handles(all_types_but_forall)
    def walk_error(self, formula: Any, **kwargs: Any) -> Any:
        return self.walk_skip(formula)

    def walk_script(self, script: SmtLibScript) -> None:
        for cmd in script.filter_by_command_name("assert"):
            self.walk(cmd.args[0])


def parse_term(parser: SmtLibParser, term: str) -> Term:
    buffer = io.StringIO(f"(assert (= {term} {term}))")

    for cmd in parser.get_command_generator(buffer):
        return cmd.args[0].args()[0]

def parse_script(smt_file: TextIO):
    parser = SmtLibParser()
    script = parser.get_script(smt_file)
    return script