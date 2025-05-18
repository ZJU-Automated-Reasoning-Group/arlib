from typing import List, Dict, Any
import json
import numpy as np
from arlib.llm.abduct.llm_abduct import AbductionProblem, AbductionResult, LLMAbductor



class AbductionEvaluator:
    """Evaluates LLM performance on abduction tasks."""
    
    def __init__(self, 
                llm_abductor: LLMAbductor,
                benchmark_problems: List[AbductionProblem]):
        """
        Initialize the evaluator.
        
        Args:
            llm_abductor: The LLM abductor to evaluate
            benchmark_problems: List of benchmark problems for evaluation
        """
        self.llm_abductor = llm_abductor
        self.benchmark_problems = benchmark_problems
        self.results = []
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the LLM abductor on all benchmark problems.
        
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        results = []
        
        for problem in self.benchmark_problems:
            result = self.llm_abductor.abduce(problem)
            results.append(result)
            
        self.results = results
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        return metrics
    
    def calculate_metrics(self, results: List[AbductionResult]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics from results.
        
        Args:
            results: List of abduction results
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        total = len(results)
        successful = sum(1 for r in results if r.is_valid)
        consistent = sum(1 for r in results if r.is_consistent)
        sufficient = sum(1 for r in results if r.is_sufficient)
        errors = sum(1 for r in results if r.error is not None)
        
        avg_time = np.mean([r.execution_time for r in results])
        
        success_rate = successful / total if total > 0 else 0
        consistency_rate = consistent / total if total > 0 else 0
        sufficiency_rate = sufficient / total if total > 0 else 0
        error_rate = errors / total if total > 0 else 0
        
        metrics = {
            "total_problems": total,
            "successful": successful,
            "consistent": consistent,
            "sufficient": sufficient,
            "errors": errors,
            "success_rate": success_rate,
            "consistency_rate": consistency_rate,
            "sufficiency_rate": sufficiency_rate,
            "error_rate": error_rate,
            "avg_execution_time": avg_time
        }
        
        return metrics
    
    def get_result_details(self) -> List[Dict[str, Any]]:
        """
        Get detailed results for all problems.
        
        Returns:
            List[Dict[str, Any]]: Detailed results
        """
        details = []
        
        for result in self.results:
            detail = {
                "problem_description": result.problem.description,
                "premise": str(result.problem.premise),
                "conclusion": str(result.problem.conclusion),
                "hypothesis": str(result.hypothesis) if result.hypothesis else None,
                "is_consistent": result.is_consistent,
                "is_sufficient": result.is_sufficient,
                "is_valid": result.is_valid,
                "execution_time": result.execution_time,
                "error": result.error
            }
            details.append(detail)
            
        return details
    
    def save_results(self, file_path: str):
        """
        Save evaluation results to a JSON file.
        
        Args:
            file_path: Path to save the results
        """
        metrics = self.calculate_metrics(self.results)
        details = self.get_result_details()
        
        data = {
            "metrics": metrics,
            "details": details
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)