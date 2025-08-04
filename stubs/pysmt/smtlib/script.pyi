from typing import Iterable, Iterator, List, Set, TextIO

from ..fnode import FNode
from .annotations import Annotations
from .printers import SmtPrinter

class SmtLibCommand:
    name: str
    args: List[FNode]

    def serialize(self, printer: SmtPrinter) -> None: ...

class SmtLibScript:
    annotations: Annotations
    commands: List[SmtLibCommand]
    special_commands: Set[int]

    def serialize(self, output: TextIO, daggify: bool = True) -> None: ...
    def add_command(self, command: SmtLibCommand) -> None: ...
    def add(self, command: str, args: List[FNode]) -> None: ...
    def filter_by_command_name(self, name: str) -> Iterator[SmtLibCommand]: ...
