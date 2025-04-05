"""
Variable
"""


class Variable:
    """
    A Boolean variable
    """

    def __init__(self, variable: int):
        self.variable_value = variable

        self.variable_abs = abs(self.variable_value)
        self.signal = 1
        if self.variable_value < 0:
            self.signal = -1

    def copy(self):
        """
        Copy the variable
        """
        return Variable(self.variable_value)

    def __str__(self):
        """
        String representation of the variable
        """
        return f"Variable({self.variable_value})"

    def __repr__(self):
        """
        Representation of the variable
        """
        return self.__str__()

    def __eq__(self, other):
        """
        Equality of the variable
        """
        return self.variable_value == other.variable_value

    def __ne__(self, other):
        """
        Inequality of the variable
        """
        return self.variable_value != other.variable_value

    def __lt__(self, other):
        """
        Less than of the variable
        """
        return self.variable_value < other.variable_value

    def __le__(self, other):
        """
        Less than or equal to of the variable
        """
        return self.variable_value <= other.variable_value

    def __gt__(self, other):
        """
        Greater than of the variable
        """
        return self.variable_value > other.variable_value

    def __ge__(self, other):
        """
        Greater than or equal to of the variable
        """
        return self.variable_value >= other.variable_value

    def __neg__(self):
        """
        Negation of the variable
        """
        return Variable(-self.variable_value)

    def __pos__(self):
        """
        Positive of the variable
        """
        return self

    def __abs__(self):
        """
        Absolute value of the variable
        """
        return Variable(abs(self.variable_value))

    def __int__(self):
        """
        Integer value of the variable
        """
        return int(self.variable_value)

    def __float__(self):
        """
        Float value of the variable
        """
        return float(self.variable_value)

    def __complex__(self):
        """
        Complex value of the variable
        """
        return complex(self.variable_value)

    def __hash__(self):
        """
        Hash of the variable
        """
        return hash(self.variable_value)
