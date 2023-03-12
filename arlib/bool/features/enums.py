from enum import Enum


class VarState(Enum):
    TRUE_VAL = 1
    FALSE_VAL = 2
    UNASSIGNED = 3
    IRRELEVANT = 4


class ClauseState(Enum):
    ACTIVE = 1
    PASSIVE = 2
