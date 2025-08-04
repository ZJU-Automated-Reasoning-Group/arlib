from typing import Optional, TextIO

from ..smtlib.annotations import Annotations

class SmtPrinter:
    def __init__(
        self, stream: TextIO, annotations: Optional[Annotations] = None
    ) -> None: ...
    def write(self, content: str) -> None: ...
