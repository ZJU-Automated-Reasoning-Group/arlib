# coding: utf-8


class Variable:
    def __init__(self, variable_string):
        if isinstance(variable_string, int):
            variable_string = str(variable_string)

        if isinstance(variable_string, Variable):
            variable_string = variable_string.original_string

        self.original_string = variable_string
        self.variable_abs = int(variable_string)
        self.variable_value = int(variable_string)
        self.signal = 1

        if self.variable_abs < 0:
            self.signal = -1
            self.variable_abs = abs(self.variable_abs)

    def copy(self):
        return Variable(self.original_string)
