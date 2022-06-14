# coding: utf-8

from .clause import Clause


class Cnf:
    def __init__(self, clause_list: list):
        self.clause_list = clause_list

    def get_clauses(self):
        """
        Return clause list
        :complexity: O(1)
        :return: clause list
        """
        return self.clause_list

    def get_number_of_literals(self):
        """
        :return: number of literals
        """
        total_set = set()
        for clause in self.get_clauses():
            total_set = total_set.union(set([abs(clause) for clause in clause.literals_set]))
        return len(total_set)

    def copy(self):
        """
        get a copy of the cnf with clauses with new ids
        :complexity: O(c*l)
        :return: copy of the cnf
        """
        new_clause_list = Cnf([clause.copy_with_new_id() for clause in self.clause_list])
        return new_clause_list

    def get_number_of_clauses(self):
        """
        :return: length of clause list
        """
        return len(self.clause_list)

    def remove_clause(self, clause):
        """
        comparison will be made with id
        :complexity: O(n)
        :param clause: clause to be removed
        """
        self.clause_list.remove(clause)

    def add_clause(self, clause):
        """
        Add a new clause to the CNF
        :complexity: O(1)
        :param clause: clause to add
        :return: add a new clause
        """
        if not isinstance(clause, Clause):
            raise Exception("Expected type Clause, and got %s" % type(clause))

        self.clause_list.append(clause)

    def get_cnf_string(self, with_zero=False):
        """
        :return: cnf string
        """
        if with_zero:
            clauses_string_list = [clause.get_clause_string(with_zero) for clause in self.clause_list]
        else:
            clauses_string_list = [clause.get_clause_string() for clause in self.clause_list]
        return '\n'.join(clauses_string_list)

    def tautology_elimination(self):
        """
        Simplify CNF by removing all clauses that are tautology
        :complexity: O(c)
        :return: a new CNF without tautological clauses
        """

        new_cnf = Cnf([])
        for clause in self.clause_list:
            if not clause.is_tautology():
                copied_clause = clause.copy_with_new_id()
                new_cnf.add_clause(copied_clause)

        return new_cnf

    def hidden_tautology_elimination(self):
        """
        Simplify CNF by removing all clauses that are hidden tautology

        :complexity: O( (c*l)^2 )
        :return: a new CNF without hidden tautological clauses
        """
        new_cnf = self.copy()

        while True:
            size = new_cnf.get_number_of_clauses()

            idx = 0
            while idx < len(new_cnf.clause_list):
                clause = new_cnf.clause_list[idx]

                hla_clause = clause.hla(new_cnf)
                is_tautology = hla_clause.is_tautology()

                if is_tautology:
                    new_cnf.remove_clause(clause)
                else:
                    idx += 1

            if new_cnf.get_number_of_clauses() == size:
                break

        return new_cnf

    def asymmetric_tautology_elimination(self):
        """
        Simplify CNF by removing all clauses that are asymmetric tautology

        :complexity: O( c^2 * l^2 * 2^l )
        :return: a new CNF without asymmetric tautological clauses
        """

        new_cnf = self.copy()

        while True:
            size = new_cnf.get_number_of_clauses()

            idx = 0
            while idx < len(new_cnf.clause_list):
                clause = new_cnf.clause_list[idx]

                ala_clause = clause.ala(new_cnf)
                is_tautology = ala_clause.is_tautology()

                if is_tautology:
                    new_cnf.remove_clause(clause)
                else:
                    idx += 1

            if new_cnf.get_number_of_clauses() == size:
                break

        return new_cnf

    def blocked_clause_elimination(self):
        """
        Simplify CNF by removing all clauses that are blocked
        :complexity: O( (c*l)^2 )
        :return: a new CNF without blocked clauses
        """

        new_cnf = self.copy()

        while True:
            size = new_cnf.get_number_of_clauses()

            idx = 0
            while idx < len(new_cnf.clause_list):
                clause = new_cnf.clause_list[idx]
                blocking_clause = clause.get_blocking_clause(new_cnf)

                if blocking_clause is not None:
                    if blocking_clause.get_set_size() < clause.get_set_size():
                        new_cnf.remove_clause(blocking_clause)
                    else:
                        new_cnf.remove_clause(clause)
                else:
                    idx += 1

            if new_cnf.get_number_of_clauses() == size:
                break

        return new_cnf

    def hidden_blocked_clause_elimination(self):
        """
        Simplify CNF by removing all clauses that are hidden blocked

        :complexity: O( (c*l)^2 )
        :return: a new CNF without hidden blocked clauses
        """

        new_cnf = self.copy()

        while True:
            size = new_cnf.get_number_of_clauses()

            idx = 0
            while idx < len(new_cnf.clause_list):
                clause = new_cnf.clause_list[idx]

                hla_clause = clause.hla(new_cnf)
                blocking_clause = hla_clause.get_blocking_clause(new_cnf)

                if blocking_clause is not None:
                    new_cnf.remove_clause(clause)
                else:
                    idx += 1

            if new_cnf.get_number_of_clauses() == size:
                break

        return new_cnf

    def asymmetric_blocked_clause_elimination(self):
        """
        Simplify CNF by removing all clauses that are asymmetric blocked

        :complexity: O( c^2 * l^2 * 2^l )
        :return: a new CNF without asymmetric blocked clauses
        """

        new_cnf = self.copy()

        while True:
            size = new_cnf.get_number_of_clauses()

            idx = 0
            while idx < len(new_cnf.clause_list):
                clause = new_cnf.clause_list[idx]

                ala_clause = clause.ala(new_cnf)
                blocking_clause = ala_clause.get_blocking_clause(new_cnf)

                if blocking_clause is not None:
                    new_cnf.remove_clause(clause)
                else:
                    idx += 1

            if new_cnf.get_number_of_clauses() == size:
                break

        return new_cnf

    def subsumption_elimination(self):
        """
        Simplify CNF by removing all clauses that are subsumed

        :complexity: O(  )

        :return: a new CNF without subsumed clauses
        """

        new_cnf = self.copy()

        while True:
            size = new_cnf.get_number_of_clauses()

            idx = 0
            while idx < len(new_cnf.clause_list):
                clause = new_cnf.clause_list[idx]

                is_subsumed = clause.is_subsumed(new_cnf)

                if is_subsumed:
                    new_cnf.remove_clause(clause)
                else:
                    idx += 1

            if new_cnf.get_number_of_clauses() == size:
                break

        return new_cnf

    def hidden_subsumption_elimination(self):
        """
        Simplify CNF by removing all clauses that are hidden subsumed
        :complexity: O( (l*c)^2 )
        :return: a new CNF without hidden subsumed clauses
        """
        new_cnf = self.copy()

        while True:
            size = new_cnf.get_number_of_clauses()

            idx = 0
            while idx < len(new_cnf.clause_list):
                clause = new_cnf.clause_list[idx]

                hla_clause = clause.hla(new_cnf)
                is_subsumed = hla_clause.is_subsumed(new_cnf)

                if is_subsumed:
                    new_cnf.remove_clause(clause)
                else:
                    idx += 1

            if new_cnf.get_number_of_clauses() == size:
                break

        return new_cnf

    def asymmetric_subsumption_elimination(self):
        """
        Simplify CNF by removing all clauses that are asymmetric subsumed

        :complexity: O( c^2 * l^2 * 2^l )
        :return: a new CNF without asymmetric subsumed clauses
        """
        new_cnf = self.copy()

        while True:
            size = new_cnf.get_number_of_clauses()

            idx = 0
            while idx < len(new_cnf.clause_list):
                clause = new_cnf.clause_list[idx]

                ala_clause = clause.ala(new_cnf)
                is_subsumed = ala_clause.is_subsumed(new_cnf)

                if is_subsumed:
                    new_cnf.remove_clause(clause)
                else:
                    idx += 1

            if new_cnf.get_number_of_clauses() == size:
                break
        return new_cnf
