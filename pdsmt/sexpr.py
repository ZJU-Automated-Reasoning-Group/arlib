# coding: utf-8
"""
Playing with sexpr
"""

# Being explicit about Types
Symbol = str
Number = (int, float)
Atom = (Symbol, Number)
List = list
Expr = (Atom, List)


def input_to_list(string: str) -> [str]:
    """
    Parse a .sl file into a list of S-Expressions.
    """
    n: int = 0
    result: [str] = []
    s: str = ""
    for c in string:
        if c == "(":
            n += 1
        if c == ")":
            n -= 1
        if c != "\n":
            s += c
        if n == 0 and s != "":
            result.append(s)
            s = ""
    return result


def tokenize(chars: str) -> list:
    """Convert a string of characters into a list of tokens."""
    return chars.replace('(', ' ( ').replace(')', ' ) ').replace('" "', 'space').split()


def parse_sexpr(program: str) -> Expr:
    """Read an S-expression from a string."""
    return read_from_tokens(tokenize(program))


def read_from_tokens(tokens: list) -> Expr:
    """Read an expression from a sequence of tokens."""
    if len(tokens) == 0:
        return
        # raise SyntaxError('unexpected EOF') # is this OK?
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)  # pop off ')'
        return L
    elif token == ')':
        raise SyntaxError('unexpected )')
    else:
        return atom(token)


def atom(token: str) -> Atom:
    """Numbers become numbers; every other token is a symbol."""
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return Symbol(token)


class ResultParser:

    def __init__(self, inputs: str):
        return

    def to_sexpr_misc(self, lines: [str]):
        """
        E.g.,
        ['and', ['=', 'x', 1], ['=', 'y', 1]]
        ['and', ['=', 'x!', ['+', 'x', 'y']], ['=', 'y!', ['+', 'x', 'y']]]
        """
        res = ["("]
        for element in lines:
            if isinstance(element, list):
                for e in self.to_sexpr_misc(element):
                    res.append(e)
            else:
                if isinstance(element, int) and self.to_real:
                    element = str(element) + ".0"
                res.append(str(element))
        res.append(")")
        return res

    def to_sexpr(self, lines: [str]):
        return " ".join(self.to_sexpr_misc(lines))


def test_parser():
    return


if __name__ == '__main__':
    test_parser()