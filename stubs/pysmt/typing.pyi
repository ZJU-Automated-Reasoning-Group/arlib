from typing import List

class PySMTType:
    param_types: List["PySMTType"]

    def is_function_type(self) -> bool: ...
    def return_type(self) -> "PySMTType": ...
