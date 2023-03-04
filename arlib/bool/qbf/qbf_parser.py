"""
Parser of QBF
"""

import re


class QBFParseError(Exception):
    """Raised when any error while parsing the input file occurs."""

    def __init__(self, msg, filename, num_lines):
        if num_lines:
            self.msg = filename + ":" + str(num_lines) + ": " + msg
        else:
            self.msg = filename + ": " + msg

    def __str__(self):
        return "[ERROR] " + self.msg


class QBFParser:
    """
    QBFParser is a parser for QBF in QDIMACS format.

    Input files to be parsed by QBFParser don't have to be strictly QDIMACS
    standard compliant, but the following basic rules have to be obeyed:
        - basic structure:
            + unspecified numer of comments of the form: c <string>
            + valid header: p cnf <pnum> <pnum> EOL
                            where pnum > 0 and EOL = '0'
            + quantifier sets, quantifiers must be either 'e' or 'a'
            + clauses
        - all input (comments, header, quantifier sets and clauses) must follow
          the grammatical structure defined in QDIMACS standard version 1.1
          (see http://www.qbflib.org/qdimacs.html#input)
        - given quantifier sets must be alternating
        - empty quantsets are not allowed
        - given number of clauses and number of clauses given must match
    Anything else not specified above may be considered as allowed and valid.
    """

    BLANK_LINE = [""]

    def parse_file(self, filename):
        """
        parse_file(filename: string)
        Parse given input file;
        return the number of variables, a list with the number of their resp.
        appearance (the so called reference count), the set of quantifier sets
        and the set of clauses if given input is valid;
        raise QBFParseError otherwise.
        """
        quantsets = []
        clauses = []
        ref_count = []

        num_quantsets = num_clauses = num_lines = 0
        pattern = re.compile("\s+")

        try:
            with open(filename, 'r') as infile:
                # comments
                for line in infile:
                    num_lines += 1
                    l = pattern.split(line.strip())
                    if l[0] != 'c' and l != self.BLANK_LINE:
                        break

                # header
                if l[0] != 'p' or l[1] != "cnf" or len(l) != 4:
                    if l[0] == 'a' or l[0] == 'e':
                        raise QBFParseError("missing header", filename,
                                            num_lines)
                    raise QBFParseError("invalid header", filename, num_lines)

                try:
                    num_vars = int(l[2])
                    if num_vars <= 0:
                        raise QBFParseError("invalid number of variables",
                                            filename, num_lines)
                    # initialise var count, ref_count[0] will never be used
                    # (all counts are directly referenced)
                    ref_count = [0 for _ in range(0, num_vars + 1)]

                    num_clauses = int(l[3])
                    if num_clauses <= 0:
                        raise QBFParseError("invalid number of clauses",
                                            filename, num_lines)
                except ValueError as err:
                    raise QBFParseError(str(err), num_lines)

                # quantifier sets
                prev_quantifier = ''

                for line in infile:
                    num_lines += 1
                    l = pattern.split(line.strip())

                    if l == self.BLANK_LINE:
                        continue

                    if l[0] != 'e' and l[0] != 'a':
                        break

                    if l[-1] != '0':
                        raise QBFParseError("missing '0' after quantset",
                                            filename, num_lines)
                    elif len(l) == 2:
                        raise QBFParseError("empty quantset", filename,
                                            num_lines)

                    quantset = [l[0]]
                    if quantset[0] == prev_quantifier:
                        raise QBFParseError("quantifiers given must be " \
                                            "alternating", filename, num_lines)
                    prev_quantifier = quantset[0]

                    try:
                        quantset.append(
                            [self._parse_literal(l_item, 1, num_vars)
                             for l_item in l[1:-1]])
                        quantsets.append(quantset)
                        num_quantsets += 1
                    except ValueError as err:
                        raise QBFParseError(str(err), filename, num_lines)
                else:
                    raise QBFParseError("missing clause definitions",
                                        filename, num_lines)
                # clauses
                if l != self.BLANK_LINE:  # first clause (already been read)
                    if l[-1] != '0':
                        raise QBFParseError("missing '0' after clause",
                                            filename, num_lines)
                    try:
                        clause = []
                        for l_item in l[:-1]:
                            lit = self._parse_literal(l_item, -num_vars,
                                                      num_vars)
                            clause.append(lit)
                            ref_count[abs(lit)] += 1
                        clauses.append(clause)

                    except ValueError as err:
                        raise QBFParseError(str(err), filename, num_lines)

                for line in infile:  # remaining clauses
                    num_lines += 1
                    l = pattern.split(line.strip())

                    if l == self.BLANK_LINE:
                        continue
                    if l[-1] != '0':
                        raise QBFParseError("missing '0' after clause",
                                            filename, num_lines)
                    try:
                        clause = []
                        for l_item in l[:-1]:
                            lit = self._parse_literal(l_item, -num_vars,
                                                      num_vars)
                            clause.append(lit)
                            ref_count[abs(lit)] += 1
                        clauses.append(clause)
                    except ValueError as err:
                        raise QBFParseError(str(err), num_lines)
                    # optimization: skip last quantset if universal

                if len(clauses) > num_clauses:
                    raise QBFParseError("too many clauses given",
                                        filename, num_lines)
                elif len(clauses) < num_clauses:
                    raise QBFParseError("not enough clauses given",
                                        filename, num_lines)
                return num_vars, ref_count, quantsets, clauses
            # end with
        except IOError as err:
            raise QBFParseError("could not open file", filename, 0)

    def _parse_literal(self, lit, range_from, range_to):
        """
        _parse_literal(lit        : string,
                       range_from : int,
                       range_to   : int)
        Convert given literal to integer and check if it is valid (within
        given range);
        return resp. integer if it is;
        raise ValueError otherwise.
        """
        try:
            value = int(lit)
        except ValueError as err:
            raise ValueError("invalid literal: '" + lit + "'")

        if value == 0:
            raise ValueError("invalid literal: '0'")

        if value < range_from or value > range_to:
            raise ValueError("invalid literal, not within range [{},{}]: " \
                             "{}".format(range_from, range_to, lit))
        return value
