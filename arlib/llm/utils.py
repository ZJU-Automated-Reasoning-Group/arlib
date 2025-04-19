"""
Utility functions for SMTO implementation.
"""

import hashlib
import json
import os
import time
import z3
from typing import Dict, Any, Optional, List


def z3_value_to_python(z3_val) -> Any:
    """
    Convert a Z3 value to its corresponding Python value.
    
    Args:
        z3_val: A Z3 value
        
    Returns:
        The corresponding Python value
    """
    if z3.is_int_value(z3_val):
        return z3_val.as_long()
    elif z3.is_real_value(z3_val):
        return float(z3_val.as_fraction())
    elif z3.is_bool_value(z3_val):
        return z3.is_true(z3_val)
    elif z3.is_string_value(z3_val):
        return z3_val.as_string()
    else:
        return str(z3_val)


def python_to_z3_value(py_val, sort: z3.SortRef):
    """
    Convert a Python value to a Z3 value of the specified sort.
    
    Args:
        py_val: A Python value
        sort: The target Z3 sort
        
    Returns:
        The corresponding Z3 value
        
    Raises:
        ValueError: If the sort is not supported
    """
    if sort == z3.IntSort():
        return z3.IntVal(py_val)
    elif sort == z3.RealSort():
        return z3.RealVal(py_val)
    elif sort == z3.BoolSort():
        return z3.BoolVal(py_val)
    elif sort == z3.StringSort():
        return z3.StringVal(py_val)
    else:
        raise ValueError(f"Unsupported sort: {sort}")


def values_equal(val1, val2) -> bool:
    """
    Check if two values are equal, handling Z3 values.
    
    Args:
        val1: First value
        val2: Second value
        
    Returns:
        True if the values are equal, False otherwise
    """
    if z3.is_expr(val1) and z3.is_expr(val2):
        return z3.eq(val1, val2)
    return val1 == val2


def generate_cache_key(oracle_name: str, inputs: Dict) -> str:
    """
    Generate a cache key for oracle inputs.
    
    Args:
        oracle_name: Name of the oracle
        inputs: Input values
        
    Returns:
        A hash-based cache key
    """
    key_data = f"{oracle_name}_{json.dumps(inputs, sort_keys=True)}"
    return hashlib.md5(key_data.encode()).hexdigest()


class OracleCache:
    """Cache for oracle query results"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the oracle cache.
        
        Args:
            cache_dir: Directory to store persistent cache, or None for in-memory only
        """
        self.cache_dir = cache_dir
        self.cache: Dict[str, Any] = {}
        
        # Load cache if cache_dir is specified
        if cache_dir and os.path.exists(cache_dir):
            self._load_cache()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value, or None if not found
        """
        return self.cache.get(key)
    
    def put(self, key: str, value: Any):
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value
        self._save_cache()
    
    def contains(self, key: str) -> bool:
        """
        Check if a key is in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key is in the cache, False otherwise
        """
        return key in self.cache
    
    def _save_cache(self):
        """Save cache to disk if cache_dir is set"""
        if not self.cache_dir:
            return
            
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Convert cache values to serializable format
        serializable_cache = {}
        for key, value in self.cache.items():
            if isinstance(value, (int, float, bool, str, type(None))):
                serializable_cache[key] = value
            else:
                serializable_cache[key] = str(value)
                
        with open(os.path.join(self.cache_dir, "oracle_cache.json"), "w") as f:
            json.dump(serializable_cache, f)

    def _load_cache(self):
        """Load cache from disk"""
        cache_file = os.path.join(self.cache_dir, "oracle_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.cache = json.load(f)


class ExplanationLogger:
    """Logger for SMTO solver explanations"""
    
    def __init__(self, level: str = "basic"):
        """
        Initialize the explanation logger.
        
        Args:
            level: Explanation level ('none', 'basic', or 'detailed')
        """
        self.level = level
        self.history: List[Dict[str, Any]] = []
    
    def log(self, message: str, level: str = "basic"):
        """
        Log an explanation message if the logger level permits.
        
        Args:
            message: The explanation message
            level: The level of this message ('basic' or 'detailed')
        """
        if self.level == "none":
            return
            
        if level == "detailed" and self.level != "detailed":
            return
            
        self.history.append({
            "timestamp": time.time(),
            "message": message,
            "level": level
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the explanation history.
        
        Returns:
            The list of explanation entries
        """
        return self.history
    
    def clear(self):
        """Clear the explanation history"""
        self.history = [] 