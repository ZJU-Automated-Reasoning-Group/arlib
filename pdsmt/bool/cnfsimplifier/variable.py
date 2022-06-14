# coding: utf-8


class Variable:
    def __init__(self, variable: int):
        self.variable_value = variable

        self.variable_abs = abs(self.variable_value)
        self.signal = 1
        if self.variable_value < 0:
            self.signal = -1

    def copy(self):
        return Variable(self.variable_value)
