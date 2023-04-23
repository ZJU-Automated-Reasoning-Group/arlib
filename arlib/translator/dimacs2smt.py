# coding: utf-8
import sys


def parse_decl(line: str, ous):
    decls = line.split()
    nvars = int(decls[2])
    for i in range(nvars):
        ous.write("(declare-const p%d Bool)\n" % (i + 1))


def parse_clause(line, ous):
    """
    Parses a single clause and writes the corresponding
    compound logic expression to `ous`.
    """
    literals = line.split()
    assert (literals[-1] == '0')
    ous.write("(assert (or ")
    for p in literals:
        if p == '0':
            break
        if p[0] == '-':
            ous.write("(not p")
            ous.write(p[1:])
            ous.write(") ")
        else:
            ous.write("p")
            ous.write(p)
            ous.write(" ")
    ous.write("))\n")


def to_qfbv(file):
    ins = open(file)
    ous = open("%s.smt2" % file, 'w')
    ous.write("(set-logic QF_BV)\n")
    line = ''
    while True:
        line = ins.readline()
        if line.startswith('p'):
            break
    parse_decl(line, ous)
    line = ins.readline()
    while line:
        parse_clause(line, ous)
        line = ins.readline()
    ous.write("(check-sat)")
    ins.close()
    ous.close()


def to_qfuf(dimacs_path: str):
    # TODO: ignore comments
    with open(dimacs_path, "r") as f:
        clauses_lines = f.read().splitlines()[1:]
        declarations = sorted(list(set(
            ["(declare-fun v_" + str(abs(int(lit))) + " () Bool)" for line in clauses_lines for lit in
             line.split(" ")[:-1]])))
        assertions = ["(assert (or " + " ".join(
            ["v_" + str(abs(int(lit))) if not lit.startswith("-") else "(not v_" + str(abs(int(lit))) + ")" for lit in
             line.split(" ")[:-1]]) + "))" for line in clauses_lines]
        content = "(set-logic QF_UF)\n" + "\n".join(declarations + assertions) + "\n(check-sat)\n"
    with open(dimacs_path + ".smt2", "w") as f:
        f.write(content)


if __name__ == '__main__':
    dimacs_path = sys.argv[1]
    # To QF_BV
    to_qfbv(dimacs_path)

    # or To QF_UF
    # to_qfuf(dimacs_path)
