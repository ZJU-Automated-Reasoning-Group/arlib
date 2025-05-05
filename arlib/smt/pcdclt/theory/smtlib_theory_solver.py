# coding: utf-8
"""
Use one or more binary solvers as the theory solver of the parallel CDCL(T) engine.

Note that we only use it for dealing with a conjunction of formulas (check theory consistency in DPLL(T))
"""
import logging
import os
import time
from typing import List, Optional, Dict
from datetime import datetime

from arlib.utils.smtlib_solver import SMTLIBSolver, SMTLIBPortfolioSolver
from arlib.smt.pcdclt.cdclt_config import (
    LOG_SMT_QUERIES, SMT_LOG_DIR, SMT_LOG_QUERY_CONTENT,
    SMT_LOG_QUERY_RESULTS, SMT_LOG_ASSUMPTIONS
)

logger = logging.getLogger(__name__)


class SMTLibTheorySolver(object):
    """
    Use smtlib_solver class to interact with a binary solver
    """

    def __init__(self, solver_bin, worker_id=None, log_dir=None):
        self.bin_solver = SMTLIBSolver(solver_bin)
        self.worker_id = worker_id
        self.log_dir = log_dir if log_dir else (SMT_LOG_DIR if LOG_SMT_QUERIES else None)
        self.query_count = 0
        self.start_time = time.time()
        self.queries_log: Dict[int, Dict] = {}
        
        # Create log directory if specified and doesn't exist
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def __del__(self):
        self.bin_solver.stop()
        # Dump all logged queries to file if log_dir is specified
        if LOG_SMT_QUERIES:
            self._dump_queries_log()

    def _log_query(self, query_type: str, content: str, assumptions: Optional[List[str]] = None, result=None):
        """
        Log a query for debugging and analysis
        """
        if not LOG_SMT_QUERIES or self.worker_id is None:
            return
            
        query_id = self.query_count
        self.query_count += 1
        
        self.queries_log[query_id] = {
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "query_type": query_type
        }
        
        # Only store content if configured to do so
        if SMT_LOG_QUERY_CONTENT:
            self.queries_log[query_id]["content"] = content
        
        # Only store assumptions if configured to do so
        if SMT_LOG_ASSUMPTIONS and assumptions:
            self.queries_log[query_id]["assumptions"] = assumptions
            
        # Only store results if configured to do so
        if SMT_LOG_QUERY_RESULTS and result is not None:
            self.queries_log[query_id]["result"] = str(result)
            
        # Also log to file if log_dir is set
        self._dump_query(query_id)
            
    def _dump_query(self, query_id: int):
        """
        Dump a single query to a file
        """
        if not self.log_dir:
            return
            
        query_data = self.queries_log[query_id]
        filename = f"worker_{self.worker_id}_query_{query_id}.smt2"
        file_path = os.path.join(self.log_dir, filename)
        
        with open(file_path, 'w') as f:
            f.write(f"; Worker ID: {self.worker_id}\n")
            f.write(f"; Query ID: {query_id}\n")
            f.write(f"; Timestamp: {query_data['timestamp']}\n")
            f.write(f"; Elapsed Time: {query_data['elapsed_time']:.6f}s\n")
            f.write(f"; Query Type: {query_data['query_type']}\n")
            
            # Write the main content if available
            if SMT_LOG_QUERY_CONTENT and "content" in query_data and query_data["content"]:
                f.write(query_data['content'])
            
            # Write assumptions if present and configured
            if SMT_LOG_ASSUMPTIONS and 'assumptions' in query_data:
                f.write("\n\n; Assumptions:\n")
                for assumption in query_data['assumptions']:
                    f.write(f"; {assumption}\n")
                
                # Also write the check-sat-assuming command
                assumptions_str = " ".join(query_data['assumptions'])
                f.write(f"\n(check-sat-assuming ({assumptions_str}))\n")
                
            # Write result if present and configured
            if SMT_LOG_QUERY_RESULTS and 'result' in query_data:
                f.write(f"\n; Result: {query_data['result']}\n")
    
    def _dump_queries_log(self):
        """
        Dump all queries to a summary file
        """
        if not self.log_dir or not self.queries_log:
            return
            
        summary_file = os.path.join(self.log_dir, f"worker_{self.worker_id}_summary.log")
        
        with open(summary_file, 'w') as f:
            f.write(f"=== Theory Solver Worker {self.worker_id} Query Summary ===\n")
            f.write(f"Total queries: {self.query_count}\n")
            f.write(f"Total time: {time.time() - self.start_time:.6f}s\n\n")
            
            for query_id, data in sorted(self.queries_log.items()):
                f.write(f"Query {query_id}:\n")
                f.write(f"  Type: {data['query_type']}\n")
                f.write(f"  Timestamp: {data['timestamp']}\n")
                f.write(f"  Elapsed: {data['elapsed_time']:.6f}s\n")
                
                if SMT_LOG_ASSUMPTIONS and 'assumptions' in data:
                    f.write(f"  Assumptions: {len(data['assumptions'])} items\n")
                
                if SMT_LOG_QUERY_RESULTS and 'result' in data:
                    f.write(f"  Result: {data['result']}\n")
                    
                f.write("\n")

    def add(self, smt2string: str):
        self._log_query("add", smt2string)
        self.bin_solver.assert_assertions(smt2string)

    def check_sat(self):
        logger.debug("Theory solver working...")
        result = self.bin_solver.check_sat()
        self._log_query("check_sat", "(check-sat)", result=result)
        return result

    def check_sat_assuming(self, assumptions: List[str]):
        """
        This is just an abstract interface
          - Some SMT solvers do not support the interface
          - We may use push/pop to simulate the behavior, or even build a solver from scratch.
        """
        logger.debug("Theory solver working...")
        # Log the query with assumptions
        self._log_query("check_sat_assuming", "", assumptions=assumptions)
        
        result = self.bin_solver.check_sat_assuming(assumptions)
        
        # Update the query log with the result
        if LOG_SMT_QUERIES and SMT_LOG_QUERY_RESULTS:
            query_id = self.query_count - 1
            if query_id in self.queries_log:
                self.queries_log[query_id]["result"] = str(result)
                self._dump_query(query_id)
            
        return result

    def get_unsat_core(self):
        core = self.bin_solver.get_unsat_core()
        self._log_query("get_unsat_core", "(get-unsat-core)", result=core)
        return core


class SMTLibPortfolioTheorySolver(object):
    """
    Use smtlib_solver class to interact with a binary solver
    TODO: test this (as we have not tried SMTLIBPortfolioSolver before)
    """

    def __init__(self, solvers_list: List[str], worker_id=None, log_dir=None):
        solvers = solvers_list
        self.bin_solvers = SMTLIBPortfolioSolver(solvers)
        self.worker_id = worker_id
        self.log_dir = log_dir if log_dir else (SMT_LOG_DIR if LOG_SMT_QUERIES else None)
        self.query_count = 0
        self.start_time = time.time()
        self.queries_log: Dict[int, Dict] = {}

        # Create log directory if specified and doesn't exist
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

    def __del__(self):
        self.bin_solvers.stop()
        # Dump all logged queries to file if log_dir is specified
        if LOG_SMT_QUERIES:
            self._dump_queries_log()

    def _log_query(self, query_type: str, content: str, assumptions: Optional[List[str]] = None, result=None):
        """
        Log a query for debugging and analysis
        """
        if not LOG_SMT_QUERIES or self.worker_id is None:
            return
            
        query_id = self.query_count
        self.query_count += 1
        
        self.queries_log[query_id] = {
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": time.time() - self.start_time,
            "query_type": query_type
        }
        
        # Only store content if configured to do so
        if SMT_LOG_QUERY_CONTENT:
            self.queries_log[query_id]["content"] = content
        
        # Only store assumptions if configured to do so
        if SMT_LOG_ASSUMPTIONS and assumptions:
            self.queries_log[query_id]["assumptions"] = assumptions
            
        # Only store results if configured to do so
        if SMT_LOG_QUERY_RESULTS and result is not None:
            self.queries_log[query_id]["result"] = str(result)
            
        # Also log to file if log_dir is set
        self._dump_query(query_id)
            
    def _dump_query(self, query_id: int):
        """
        Dump a single query to a file
        """
        if not self.log_dir:
            return
            
        query_data = self.queries_log[query_id]
        filename = f"portfolio_worker_{self.worker_id}_query_{query_id}.smt2"
        file_path = os.path.join(self.log_dir, filename)
        
        with open(file_path, 'w') as f:
            f.write(f"; Worker ID: {self.worker_id} (Portfolio)\n")
            f.write(f"; Query ID: {query_id}\n")
            f.write(f"; Timestamp: {query_data['timestamp']}\n")
            f.write(f"; Elapsed Time: {query_data['elapsed_time']:.6f}s\n")
            f.write(f"; Query Type: {query_data['query_type']}\n")
            
            # Write the main content if available
            if SMT_LOG_QUERY_CONTENT and "content" in query_data and query_data["content"]:
                f.write(query_data['content'])
            
            # Write assumptions if present and configured
            if SMT_LOG_ASSUMPTIONS and 'assumptions' in query_data:
                f.write("\n\n; Assumptions:\n")
                for assumption in query_data['assumptions']:
                    f.write(f"; {assumption}\n")
                
                # Also write the check-sat-assuming command
                assumptions_str = " ".join(query_data['assumptions'])
                f.write(f"\n(check-sat-assuming ({assumptions_str}))\n")
                
            # Write result if present and configured
            if SMT_LOG_QUERY_RESULTS and 'result' in query_data:
                f.write(f"\n; Result: {query_data['result']}\n")
    
    def _dump_queries_log(self):
        """
        Dump all queries to a summary file
        """
        if not self.log_dir or not self.queries_log:
            return
            
        summary_file = os.path.join(self.log_dir, f"portfolio_worker_{self.worker_id}_summary.log")
        
        with open(summary_file, 'w') as f:
            f.write(f"=== Portfolio Theory Solver Worker {self.worker_id} Query Summary ===\n")
            f.write(f"Total queries: {self.query_count}\n")
            f.write(f"Total time: {time.time() - self.start_time:.6f}s\n\n")
            
            for query_id, data in sorted(self.queries_log.items()):
                f.write(f"Query {query_id}:\n")
                f.write(f"  Type: {data['query_type']}\n")
                f.write(f"  Timestamp: {data['timestamp']}\n")
                f.write(f"  Elapsed: {data['elapsed_time']:.6f}s\n")
                
                if SMT_LOG_ASSUMPTIONS and 'assumptions' in data:
                    f.write(f"  Assumptions: {len(data['assumptions'])} items\n")
                
                if SMT_LOG_QUERY_RESULTS and 'result' in data:
                    f.write(f"  Result: {data['result']}\n")
                    
                f.write("\n")

    def add(self, smt2string: str):
        self._log_query("add", smt2string)
        self.bin_solvers.assert_assertions(smt2string)

    def check_sat(self):
        logger.debug("Theory solver working...")
        result = self.bin_solvers.check_sat()
        self._log_query("check_sat", "(check-sat)", result=result)
        return result

    def check_sat_assuming(self, assumptions: List[str]):
        """
        This is just an abstract interface
          - Some SMT solvers do not support the interface
          - We may use push/pop to simulate the behavior, or even build a solver from scratch.
        """
        logger.debug("Theory solver working...")
        # Log query with assumptions
        self._log_query("check_sat_assuming", "", assumptions=assumptions)
        
        cnts = "(assert ( and {}))\n".format(" ".join(assumptions))
        self.add(cnts)
        result = self.check_sat()
        
        # Update the query log with the result
        if LOG_SMT_QUERIES and SMT_LOG_QUERY_RESULTS:
            query_id = self.query_count - 1
            if query_id in self.queries_log:
                self.queries_log[query_id]["result"] = str(result)
                self._dump_query(query_id)
            
        return result

    def get_unsat_core(self):
        core = self.bin_solvers.get_unsat_core()
        self._log_query("get_unsat_core", "(get-unsat-core)", result=core)
        return core
