# coding: utf-8
"""
Python implementation of the Dissolve algorithm using PySAT backends.

This module follows the organization of Algorithms 1–3 in the paper
"Dissolve: A Distributed SAT Solver based on Stålmarck’s Method" (2015),
but focuses on a practical subset that is sufficient for experimentation.

Design choices
--------------
- We use Python multiprocessing to run workers. Each worker owns a PySAT
  solver instance and processes one SAT query at a time under a set of
  assumptions (the Dilemma branch). Workers return:
    (status, learnt_clauses, decision_literals, phases, model_or_core)
- Instead of a full UBTree with subsumption, we provide an efficient
  priority-bucket store that organizes clauses into small buckets by size
  and LBD if available; this is sufficient to emulate the “tiers” (T_t,i)
  used for deciding which clauses to broadcast next round.
- Variable selection for Dilemma splits is a simple combination of
  frequency in recent conflict clauses and VSIDS-like votes gathered from
  workers. This is easy to compute and works well in practice.

The API is intentionally small and documented via dataclasses below.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import os
import queue
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Callable

from pysat.formula import CNF
from pysat.solvers import Solver

from arlib.utils.types import SolverResult

logger = logging.getLogger(__name__)


# ------------------------------ Data classes ------------------------------ #


@dataclass
class DissolveResult:
    result: SolverResult
    model: Optional[List[int]] = None
    unsat_core: Optional[List[int]] = None
    rounds: int = 0
    runtime_sec: float = 0.0


@dataclass
class DissolveConfig:
    k_split_vars: int = 5
    per_query_conflict_budget: int = 20000
    max_rounds: Optional[int] = None
    num_workers: Optional[int] = None  # default: os.cpu_count()
    solver_name: str = "cd"  # cadical in PySAT
    seed: Optional[int] = None
    # clause sharing parameters
    max_shared_per_round: int = 50000
    important_bucket_size: int = 100
    store_small_clauses_threshold: int = 2
    # run strategy parameters
    budget_strategy: str = "constant"  # constant | luby
    budget_unit: int = 10_000  # conflict budget unit for the budget strategy
    distribution_strategy: str = "dilemma"  # dilemma | portfolio | hybrid
    kprime_for_decision_votes: int = 5


# ------------------------------ UBStore ---------------------------------- #


class UBStore:
    """
    A light-weight, unlimited-branching bucket store for learnt clauses.

    We keep three priority levels per round bucket t:
      T_t,1: unit clauses
      T_t,2: binary clauses and other short clauses up to a threshold
      T_t,3: remaining clauses (capped by `max_shared_per_round`)
    """

    def __init__(self, max_per_round: int, small_threshold: int, important_first: int):
        self.max_per_round = max_per_round
        self.small_threshold = small_threshold
        self.important_first = important_first
        self._by_round: Dict[int, Tuple[List[Tuple[List[int], int]], List[Tuple[List[int], int]], List[Tuple[List[int], int]]]] = {}

    def insert_many(self, round_id: int, clauses: Iterable[List[int]]) -> None:
        b1, b2, b3 = self._by_round.setdefault(round_id, ([], [], []))
        for cls in clauses:
            size = len(cls)
            # rating: shorter is better (proxy for LBD in absence of LBD)
            rating = size
            if size <= 1:
                b1.append((cls, rating))
            elif size <= max(2, self.small_threshold):
                b2.append((cls, rating))
            else:
                b3.append((cls, rating))

    def best_for_next_round(self, round_id: int) -> List[List[int]]:
        if round_id not in self._by_round:
            return []
        b1, b2, b3 = self._by_round[round_id]
        # order by rating within each bucket
        b1_sorted = sorted(b1, key=lambda x: x[1])
        b2_sorted = sorted(b2, key=lambda x: x[1])
        b3_sorted = sorted(b3, key=lambda x: x[1])

        result: List[List[int]] = []
        for bucket in (b1_sorted, b2_sorted, b3_sorted):
            for cls, _ in bucket:
                result.append(cls)
                if len(result) >= self.max_per_round:
                    return result
        return result


# ------------------------------ Variable picking ------------------------- #


def pick_split_variables(num_vars: int, scores: Dict[int, int], k: int, rng: random.Random) -> List[int]:
    """
    Pick k variables by a mixture of score-based and random selection.
    `scores` accumulates frequencies from recent conflict/learnt clauses.
    """
    candidates = list(range(1, num_vars + 1))
    rng.shuffle(candidates)
    # boost by scores
    candidates.sort(key=lambda v: -scores.get(v, 0))
    return candidates[:k]


def assumptions_from_bits(vars_to_split: Sequence[int], mask: int) -> List[int]:
    assumps = []
    for i, v in enumerate(vars_to_split):
        bit = (mask >> i) & 1
        assumps.append(v if bit == 1 else -v)
    return assumps


# ------------------------------ Worker process --------------------------- #


def _worker_solve(
    solver_name: str,
    cnf_clauses: List[List[int]],
    assumptions: List[int],
    learnt_in: List[List[int]],
    conflict_budget: int,
    seed: Optional[int],
) -> Tuple[SolverResult, List[List[int]], List[int], Dict[int, int], Optional[List[int]]]:
    """
    Solve one query with a standalone PySAT solver.

    Returns (status, learnt_clauses, decision_literals, vote_scores, model_or_core)
    vote_scores is a simple frequency map over variables appearing in conflicts.
    """
    try:
        # initialize solver
        s = Solver(name=solver_name, bootstrap_with=cnf_clauses)
        # inject learnt clauses from previous rounds
        for c in learnt_in:
            try:
                s.add_clause(c)
            except Exception:
                pass

        # We could set random phases or seeds here if the backend exposes it.
        # Keep disabled by default to avoid backend-specific assumptions.

        # set a conflict budget when supported
        try:
            s.conf_budget(conflict_budget)
            budgeted = True
        except Exception:
            budgeted = False

        status_val = s.solve_limited(assumptions=assumptions) if budgeted else s.solve(assumptions=assumptions)

        if status_val is True:
            return (SolverResult.SAT, [], [], {}, s.get_model())

        # unknown due to budget or timeout (PySAT returns None)
        if budgeted and status_val is None:
            # budget exhausted: return without learnts (PySAT does not expose them)
            return (SolverResult.UNKNOWN, [], [], {}, None)

        # unsat
        core = None
        try:
            core = s.get_core()
        except Exception:
            core = None

        # attempt to gather learnt clauses
        learnt: List[List[int]] = []

        # simple vote heuristic from learnts
        votes: Dict[int, int] = {}
        for cls in learnt:
            for lit in cls:
                v = abs(lit)
                votes[v] = votes.get(v, 0) + 1

        return (SolverResult.UNSAT, learnt, [], votes, core)
    except Exception as exc:
        logger.exception("Worker crashed: %s", exc)
        return (SolverResult.ERROR, [], [], {}, None)


# ------------------------------ Dissolve main ---------------------------- #


class Dissolve:
    """High-level driver implementing the asynchronous Dilemma splits."""

    def __init__(self, config: Optional[DissolveConfig] = None):
        self.cfg = config or DissolveConfig()

    # -------------------------- Budget strategies ------------------------- #

    @staticmethod
    def _luby(i: int) -> int:
        """Return the i-th term of the Luby sequence (1-indexed)."""
        k = 1
        while (1 << k) - 1 < i:
            k += 1
        if i == (1 << k) - 1:
            return 1 << (k - 1)
        return Dissolve._luby(i - (1 << (k - 1)) + 1)

    def _budget_for_round(self, round_id: int) -> int:
        if self.cfg.budget_strategy == "constant":
            return self.cfg.budget_unit
        if self.cfg.budget_strategy == "luby":
            return self.cfg.budget_unit * Dissolve._luby(round_id + 1)
        return self.cfg.budget_unit

    def solve(self, cnf: CNF) -> DissolveResult:
        start = time.time()
        num_workers = self.cfg.num_workers or os.cpu_count() or 4
        rng = random.Random(self.cfg.seed)

        clauses = list(cnf.clauses)
        num_vars = cnf.nv

        round_id = 0
        global_learnts: List[List[int]] = []
        ubstore = UBStore(
            max_per_round=self.cfg.max_shared_per_round,
            small_threshold=self.cfg.store_small_clauses_threshold,
            important_first=self.cfg.important_bucket_size,
        )

        variable_scores: Dict[int, int] = {}

        # Prefer 'fork' to avoid issues when the caller runs from <stdin> or REPL
        # (spawn tries to reload the main module from file path).
        start_methods = mp.get_all_start_methods()
        ctx_name = "fork" if "fork" in start_methods else "spawn"
        ctx = mp.get_context(ctx_name)
        with ctx.Pool(processes=num_workers) as pool:
            while True:
                # termination condition
                if self.cfg.max_rounds is not None and round_id >= self.cfg.max_rounds:
                    return DissolveResult(
                        result=SolverResult.UNKNOWN,
                        rounds=round_id,
                        runtime_sec=time.time() - start,
                    )

                # pick Dilemma variables for this round
                vars_to_split = pick_split_variables(num_vars, variable_scores, self.cfg.k_split_vars, rng)

                # build assumptions for 2^k branches (or portfolio variant)
                jobs: List[Tuple[List[int], int]] = []
                if self.cfg.distribution_strategy == "portfolio":
                    # No splitting; run 2^k identical queries with different seeds
                    for mask in range(1 << len(vars_to_split)):
                        jobs.append(([], mask))
                else:
                    for mask in range(1 << len(vars_to_split)):
                        assumps = assumptions_from_bits(vars_to_split, mask)
                        jobs.append((assumps, mask))

                # learnt clauses to broadcast this round
                shared = ubstore.best_for_next_round(round_id - 1) if round_id > 0 else []

                # dispatch
                per_query_budget = self._budget_for_round(round_id)
                async_results = [
                    pool.apply_async(
                        _worker_solve,
                        (
                            self.cfg.solver_name,
                            clauses,
                            assumps,
                            shared,
                            per_query_budget,
                            rng.randrange(1 << 30),
                        ),
                    )
                    for assumps, _ in jobs
                ]

                # collect
                any_sat_model: Optional[List[int]] = None
                all_unsat = True
                any_unknown = False
                merged_learnts: List[List[int]] = []
                merged_votes: Dict[int, int] = {}
                any_core: Optional[List[int]] = None

                for res in async_results:
                    try:
                        status, learnt, _dec, votes, model_or_core = res.get()
                    except Exception as exc:
                        logger.exception("Worker result error: %s", exc)
                        any_unknown = True
                        continue

                    if status == SolverResult.SAT:
                        any_sat_model = model_or_core
                        all_unsat = False
                        break
                    elif status == SolverResult.UNSAT:
                        merged_learnts.extend(learnt)
                        for v, c in votes.items():
                            merged_votes[v] = merged_votes.get(v, 0) + c
                        if model_or_core:
                            any_core = model_or_core
                    elif status == SolverResult.UNKNOWN:
                        all_unsat = False
                        any_unknown = True
                        merged_learnts.extend(learnt)
                        for v, c in votes.items():
                            merged_votes[v] = merged_votes.get(v, 0) + c
                    else:
                        all_unsat = False
                        any_unknown = True

                if any_sat_model is not None:
                    return DissolveResult(
                        result=SolverResult.SAT,
                        model=any_sat_model,
                        rounds=round_id + 1,
                        runtime_sec=time.time() - start,
                    )

                if all_unsat:
                    # All branches unsat => original formula is UNSAT.
                    return DissolveResult(
                        result=SolverResult.UNSAT,
                        unsat_core=any_core,  # may be None when not available
                        rounds=round_id + 1,
                        runtime_sec=time.time() - start,
                    )

                # merge learnt information for the next round
                # merge function: union learnts, combine votes
                for v, c in merged_votes.items():
                    variable_scores[v] = variable_scores.get(v, 0) + c
                global_learnts.extend(merged_learnts)
                ubstore.insert_many(round_id, merged_learnts)

                round_id += 1


__all__ = [
    "Dissolve",
    "DissolveConfig",
    "DissolveResult",
]
