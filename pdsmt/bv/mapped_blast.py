#!/usr/bin/env python3

# remember to install z3 (or put the 'build' dir in the PYTHONPATH
# var)

from z3 import *
from z3.z3util import get_vars


# p cnf nvar nclauses
# cr projected_var_ids


def bitblast(formula):
    # input_vars = [x for x in collect_vars(formula)] # this might be slow?
    input_vars = get_vars(formula)
    # map bits in the bv input vars
    map_clauses, map_vars, bv2bool = map_bitvector(input_vars)
    # print(bv2bool)
    id_table = {}  # {varname: id}

    curr_id = 1
    for var in map_vars:
        name = var.decl().name()
        id_table[name] = curr_id
        curr_id = curr_id + 1
    # projection_scope = curr_id - 1
    # bitblast
    g = Goal()
    g.add(map_clauses)
    g.add(formula)
    # t = Then('simplify', 'bit-blast', 'tseitin-cnf')
    t = Then('simplify', 'bit-blast', 'tseitin-cnf')
    bitblasted = t(g)[0]

    return bitblasted, id_table, bv2bool


def is_literal(exp):
    return is_const(exp) and exp.decl().kind() == Z3_OP_UNINTERPRETED


def is_ite(exp):
    return exp.decl().kind() == Z3_OP_ITE


def is_iff(exp):
    return exp.decl().kind() == Z3_OP_IFF


def proj_id_last(var, n_proj_vars, n_vars):
    assert var != 0
    is_neg = var < 0
    if abs(var) <= n_proj_vars:
        new_var = abs(var) - n_proj_vars + n_vars
    else:
        new_var = abs(var) - n_proj_vars

    return new_var * (-1 if is_neg else 1)


def to_dimacs(cnf, table, proj_last):
    cnf_clauses = []
    projection_scope = len(table)

    for clause in cnf:
        # print(clause)
        assert is_or(clause) or is_not(clause) or is_literal(clause)
        dimacs_clause = list(dimacs_visitor(clause, table))
        # dimacs_clause.append('0') # TODO: why append 0???
        cnf_clauses.append(" ".join(dimacs_clause))

    if proj_last:
        n_vars = len(table)
        clauses = []
        for clause in cnf_clauses:
            int_clause = [int(x) for x in clause.split(" ")[:-1]]
            proj_clause = [proj_id_last(x, projection_scope, n_vars) for x in int_clause]
            # proj_clause.append(0)  # TODO: why append 0???
            str_clause = " ".join([str(x) for x in proj_clause])
            clauses.append(str_clause)

        cnf_clauses = clauses
        cnf_header = [
            "p cnf {} {}".format(len(table), len(cnf_clauses)),
            "cr {}".format(" ".join([str(x) for x in range(n_vars - projection_scope + 1, n_vars + 1)]))
        ]
    else:
        cnf_header = [
            "p cnf {} {}".format(len(table), len(cnf_clauses)),
            "cr {}".format(" ".join([str(x) for x in range(1, projection_scope + 1)]))
        ]
    return cnf_header, cnf_clauses


def map_bitvector(input_vars):
    clauses = []
    mapped_vars = []
    bv2bool = {}  # for tracking what Bools corresponding to a bv
    for var in input_vars:
        name = var.decl().name()
        size = var.size()
        boolvars = []
        for x in range(size):
            extracted_bool = Bool(name + "!" + str(x))
            clause = extracted_bool == (Extract(x, x, var) == BitVecVal(1, 1))  # why adding this
            mapped_vars.append(extracted_bool)
            clauses.append(clause)
            boolvars.append(name + "!" + str(x))

        bv2bool[str(name)] = boolvars
    return clauses, mapped_vars, bv2bool


hit = 0
miss = 0


def bexpr_visitor(exp, table, cache):
    global hit, miss
    if exp in cache:
        hit += 1
        print("hit: {}".format(hit))
        yield cache[exp]
        return
    else:
        miss += 1
        print("miss: {}".format(miss))
        if is_literal(exp):
            name = exp.decl().name().replace('!', "X")
            if name in table:
                bvar = table[name]
            else:
                bvar = bexpr.exprvar(name)
                table[name] = bvar
            cache[exp] = bvar
            yield bvar
            return
        elif is_not(exp):
            assert len(exp.children()) == 1
            ch = exp.children()[0]
            for var in bexpr_visitor(ch, table, cache):
                term = bexpr.Not(var, simplify=False)
                cache[exp] = term
                yield term
            return
        elif is_or(exp):
            or_clauses = []
            for ch in exp.children():
                # assert is_not(ch) or is_literal(ch)
                for var in bexpr_visitor(ch, table, cache):
                    or_clauses.append(var)
            term = bexpr.Or(*or_clauses, simplify=False)
            cache[exp] = term
            yield term
            return
        elif is_eq(exp) or is_iff(exp):
            eq_clauses = []
            for ch in exp.children():
                for var in bexpr_visitor(ch, table, cache):
                    eq_clauses.append(var)
            term = bexpr.Equal(*eq_clauses, simplify=False)
            cache[exp] = term
            yield term
            return
        elif is_ite(exp):
            children = exp.children()
            assert len(children) == 3
            ite_clauses = []
            for ch in children:
                for var in bexpr_visitor(ch, table, cache):
                    ite_clauses.append(var)
            assert len(ite_clauses) == 3
            # in pyEDA: ITE(expr,if_false,if_true)
            term = bexpr.ITE(ite_clauses[0], ite_clauses[2], ite_clauses[1])
            cache[exp] = term
            yield term
            return
        else:
            raise Exception("Unhandled type: ", exp)


def dimacs_visitor(exp, table):
    if is_literal(exp):
        name = exp.decl().name()
        if name in table:
            id_var = table[name]
        else:
            id_var = len(table) + 1
            table[name] = id_var
        yield str(id_var)
        return
    elif is_not(exp):
        assert len(exp.children()) == 1
        ch = exp.children()[0]
        for var in dimacs_visitor(ch, table):
            yield "-" + var
        return
    elif is_or(exp):
        for ch in exp.children():
            for var in dimacs_visitor(ch, table):
                yield var
        return
    else:
        if is_true(exp): return  # corrent?
        # elif is_false(e): return ??
        raise Exception("Unhandled type: ", exp)


def collect_vars(exp, seen=None):
    if seen is None:
        seen = {}
    if exp in seen:
        return
    seen[exp] = True

    # check if 'e' is a bitvector input variable
    if is_literal(exp) and is_bv(exp):
        yield exp
    elif is_app(exp):
        for ch in exp.children():
            for exp in collect_vars(ch, seen):
                yield exp
        return
    elif is_quantifier(exp):
        for exp in collect_vars(exp.body(), seen):
            yield exp
        return


def translate_smt2formula_to_cnf(formula):
    projection_last = ''
    projection_last = projection_last and projection_last.lower() != "false"
    # print("Generating DIMACS with projection...")
    bitblasted, id_table, bv2bool = bitblast(formula)
    # FIXME: if formula is very simple, the "simplify" tactic may convert it to "False",
    # then to_dimacs may issue a warning
    # FIXME: but it seems that cnf tactic needs "simplify" to perform some normlization
    # FIXME: maybe we need to turn the parameters of "simplify"?
    header, clauses = to_dimacs(bitblasted, id_table, projection_last)
    # print(bitblasted)
    return bv2bool, id_table, header, clauses


# TODO: what is projection_last
def test(inputfile):
    projection_last = ''
    projection_last = projection_last and projection_last.lower() != "false"

    formula_vec = parse_smt2_file(inputfile)
    formula = And(formula_vec)
    print("Generating DIMACS with projection...")
    bitblasted, id_table, bv2bool = bitblast(formula)
    header, clauses = to_dimacs(bitblasted, id_table, projection_last)
    # print('\n'.join(header))
    print('\n'.join(clauses))
    # print(id_table)
