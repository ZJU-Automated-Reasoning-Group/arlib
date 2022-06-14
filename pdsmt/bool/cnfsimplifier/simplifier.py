# coding: utf-8

"""
Conjunctive Normal Form expression simplifier that preserves satisfiability
"""


def tautoly(cnf):
    cnf = cnf.tautology_elimination()
    return cnf


def blocked_clause(cnf):
    cnf = cnf.blocked_clause_elimination()
    return cnf


def subsumption_elimination(cnf):
    cnf = cnf.subsumption_elimination()
    return cnf


def hidden_tautoly(cnf):
    cnf = cnf.hidden_tautology_elimination()
    return cnf


def hidden_blocked_clause(cnf):
    cnf = cnf.hidden_blocked_clause_elimination()
    return cnf


def hidden_subsumption_elimination(cnf):
    cnf = cnf.hidden_subsumption_elimination()
    return cnf


def asymmetric_tautoly(cnf):
    cnf = cnf.asymmetric_tautology_elimination()
    return cnf


def asymmetric_blocked_clause(cnf):
    cnf = cnf.asymmetric_blocked_clause_elimination()
    return cnf


def asymmetric_subsumption_elimination(cnf):
    cnf = cnf.asymmetric_subsumption_elimination()
    return cnf


def explicits(cnf):
    cnf = cnf.subsumption_elimination()
    cnf = cnf.blocked_clause_elimination()
    cnf = cnf.tautology_elimination()
    return cnf


def hiddens(cnf):
    cnf = cnf.hidden_subsumption_elimination()
    cnf = cnf.hidden_tautology_elimination()
    cnf = cnf.hidden_blocked_clause_elimination()
    return cnf


def asymmetrics(cnf):
    cnf = cnf.asymmetric_subsumption_elimination()
    cnf = cnf.asymmetric_tautology_elimination()
    cnf = cnf.asymmetric_blocked_clause_elimination()
    return cnf


def complete(cnf):
    cnf = cnf.asymmetric_subsumption_elimination()

    cnf = cnf.blocked_clause_elimination()
    cnf = cnf.tautology_elimination()
    cnf = cnf.subsumption_elimination()

    # cnf = cnf.hidden_tautology_elimination()
    cnf = cnf.asymmetric_tautology_elimination()

    # cnf = cnf.hidden_blocked_clause_elimination()
    cnf = cnf.asymmetric_blocked_clause_elimination()

    # cnf = cnf.hidden_subsumption_elimination()
    # cnf = cnf.asymmetric_subsumption_elimination()

    return cnf
