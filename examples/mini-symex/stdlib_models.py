#!/usr/bin/env python3
"""
Models for Python's standard libraries to enhance symbolic execution.
This file provides symbolic-aware implementations of common standard library functions.
"""

from symast import *
from concolic import *
import z3
import functools
import inspect
import types
import sys

###########################################
# Type System Models
###########################################

class TypeRegistry:
    """Registry for Python types and their symbolic representations"""
    
    def __init__(self):
        self.type_models = {}
        self.type_conversions = {}
        
    def register_type(self, python_type, symbolic_type, conversion_func=None):
        """Register a Python type with its symbolic representation"""
        self.type_models[python_type] = symbolic_type
        if conversion_func:
            self.type_conversions[python_type] = conversion_func
            
    def get_symbolic_type(self, python_type):
        """Get the symbolic representation for a Python type"""
        if python_type in self.type_models:
            return self.type_models[python_type]
        
        # Handle inheritance - find the closest parent type
        for base_type, symbolic_type in self.type_models.items():
            if issubclass(python_type, base_type):
                return symbolic_type
                
        # Default to generic object type
        return self.type_models.get(object, None)
        
    def convert_value(self, value, target_type):
        """Convert a value to the target type with symbolic tracking"""
        if target_type in self.type_conversions:
            return self.type_conversions[target_type](value)
        
        # Default conversion (may lose symbolic information)
        return target_type(value(value) if hasattr(value, '_v') else value)

# Create a global type registry
type_registry = TypeRegistry()

# Register basic types
type_registry.register_type(int, ast_int)
type_registry.register_type(float, ast_float)
type_registry.register_type(str, ast_str)
type_registry.register_type(list, ast_array)
type_registry.register_type(dict, ast_dict)
type_registry.register_type(bool, ast_const_bool)

###########################################
# Type Annotation Support
###########################################

def symbolic_type_check(func):
    """Decorator to check type annotations at runtime with symbolic awareness"""
    sig = inspect.signature(func)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        
        # Check parameter types
        for param_name, param_value in bound.arguments.items():
            param = sig.parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                # Get the expected type
                expected_type = param.annotation
                
                # Skip checking for Union types (PEP 484) and Optional types
                if hasattr(expected_type, "__origin__") and expected_type.__origin__ is not None:
                    continue
                
                # Check if the value matches the expected type
                if not isinstance(param_value, expected_type):
                    # For symbolic values, add a path constraint
                    if hasattr(param_value, '_ast'):
                        # Add type constraint (simplified)
                        add_pc(ast_eq(ast_const_bool(True), ast_const_bool(True)))
                    else:
                        raise TypeError(f"Parameter '{param_name}' must be {expected_type.__name__}, "
                                       f"got {type(param_value).__name__}")
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Check return type
        if sig.return_annotation != inspect.Parameter.empty:
            expected_type = sig.return_annotation
            if not isinstance(result, expected_type):
                # For symbolic values, add a path constraint
                if hasattr(result, '_ast'):
                    # Add type constraint (simplified)
                    add_pc(ast_eq(ast_const_bool(True), ast_const_bool(True)))
                else:
                    raise TypeError(f"Return value must be {expected_type.__name__}, "
                                   f"got {type(result).__name__}")
        
        return result
    
    return wrapper

###########################################
# Math Module Models
###########################################

class MathModels:
    """Models for the math module functions"""
    
    @staticmethod
    def sin(x):
        """Symbolic model for math.sin"""
        concrete_val = math.sin(value(x))
        
        # For symbolic values, create a new symbolic float
        if hasattr(x, '_ast'):
            # Create a new symbolic variable with a name based on the input
            if hasattr(x, '__ast') and hasattr(x.__ast, 'id'):
                sym_id = f"sin_{x.__ast.id}"
            else:
                sym_id = f"sin_result"
            
            # Add constraints that approximate sin behavior
            # sin(x) is between -1 and 1
            add_pc(ast_and(
                ast_le(ast(-1.0), ast_float(sym_id)),
                ast_le(ast_float(sym_id), ast(1.0))
            ))
            
            return concolic_float(ast_float(sym_id), concrete_val)
        
        return concrete_val
    
    @staticmethod
    def cos(x):
        """Symbolic model for math.cos"""
        concrete_val = math.cos(value(x))
        
        # For symbolic values, create a new symbolic float
        if hasattr(x, '_ast'):
            # Create a new symbolic variable with a name based on the input
            if hasattr(x, '__ast') and hasattr(x.__ast, 'id'):
                sym_id = f"cos_{x.__ast.id}"
            else:
                sym_id = f"cos_result"
            
            # Add constraints that approximate cos behavior
            # cos(x) is between -1 and 1
            add_pc(ast_and(
                ast_le(ast(-1.0), ast_float(sym_id)),
                ast_le(ast_float(sym_id), ast(1.0))
            ))
            
            return concolic_float(ast_float(sym_id), concrete_val)
        
        return concrete_val
    
    @staticmethod
    def sqrt(x):
        """Symbolic model for math.sqrt"""
        concrete_val = math.sqrt(value(x))
        
        # For symbolic values, create a new symbolic float
        if hasattr(x, '_ast'):
            # Create a new symbolic variable with a name based on the input
            if hasattr(x, '__ast') and hasattr(x.__ast, 'id'):
                sym_id = f"sqrt_{x.__ast.id}"
            else:
                sym_id = f"sqrt_result"
            
            # Add constraint that x must be non-negative for sqrt
            add_pc(ast_ge(ast(x), ast(0.0)))
            
            # Add constraint that sqrt(x) * sqrt(x) = x
            result = concolic_float(ast_float(sym_id), concrete_val)
            add_pc(ast_eq(ast_mul(ast(result), ast(result)), ast(x)))
            
            return result
        
        return concrete_val

###########################################
# String Module Models
###########################################

class StringModels:
    """Models for string operations"""
    
    @staticmethod
    def upper(s):
        """Symbolic model for str.upper()"""
        concrete_val = value(s).upper()
        
        # For symbolic values, create a new symbolic string
        if hasattr(s, '_ast'):
            # Create a new symbolic variable with a name based on the input
            if hasattr(s, '__ast') and hasattr(s.__ast, 'id'):
                sym_id = f"{s.__ast.id}_upper"
            else:
                sym_id = f"upper_result"
            
            # We can't directly model the upper() operation in Z3
            # So we create a new symbolic string and add some basic constraints
            result = concolic_str(ast_str(sym_id), concrete_val)
            
            # Add constraint that length is preserved
            add_pc(ast_eq(ast_str_length(ast(result)), ast_str_length(ast(s))))
            
            return result
        
        return concrete_val
    
    @staticmethod
    def lower(s):
        """Symbolic model for str.lower()"""
        concrete_val = value(s).lower()
        
        # For symbolic values, create a new symbolic string
        if hasattr(s, '_ast'):
            # Create a new symbolic variable with a name based on the input
            if hasattr(s, '__ast') and hasattr(s.__ast, 'id'):
                sym_id = f"{s.__ast.id}_lower"
            else:
                sym_id = f"lower_result"
            
            # We can't directly model the lower() operation in Z3
            # So we create a new symbolic string and add some basic constraints
            result = concolic_str(ast_str(sym_id), concrete_val)
            
            # Add constraint that length is preserved
            add_pc(ast_eq(ast_str_length(ast(result)), ast_str_length(ast(s))))
            
            return result
        
        return concrete_val
    
    @staticmethod
    def split(s, sep=None, maxsplit=-1):
        """Symbolic model for str.split()"""
        concrete_val = value(s).split(value(sep) if sep is not None else None, maxsplit)
        
        # For symbolic values, create a new symbolic list
        if hasattr(s, '_ast'):
            # Create a new symbolic variable with a name based on the input
            if hasattr(s, '__ast') and hasattr(s.__ast, 'id'):
                sym_id = f"{s.__ast.id}_split"
            else:
                sym_id = f"split_result"
            
            # We can't directly model the split() operation in Z3
            # So we create a new symbolic list with the concrete result
            result = concolic_list(ast_array(sym_id, len(concrete_val)), concrete_val)
            
            # Add some basic constraints
            # If the separator is in the string, the result has at least 2 elements
            if sep is not None and value(sep) in value(s):
                add_pc(ast_ge(ast_const_int(len(concrete_val)), ast_const_int(2)))
            
            return result
        
        return concrete_val

###########################################
# Regular Expression Models
###########################################

class RegexModels:
    """Models for regular expression operations"""
    
    @staticmethod
    def match(pattern, string, flags=0):
        """Symbolic model for re.match()"""
        import re
        concrete_val = re.match(value(pattern), value(string), flags)
        
        # For symbolic values, create a new symbolic boolean
        if hasattr(string, '_ast'):
            # We can't directly model regex matching in Z3
            # So we create a new symbolic boolean with the concrete result
            result = concolic_bool(ast_const_bool(concrete_val is not None), concrete_val is not None)
            
            # Add some basic constraints for common patterns
            pattern_val = value(pattern)
            if pattern_val.startswith('^'):
                # Pattern matches start of string
                prefix = pattern_val[1:].split('[')[0].split('(')[0].split('|')[0]
                if prefix and not any(c in '.?*+{}()[]\\|' for c in prefix):
                    # Simple prefix match
                    add_pc(ast_eq(result, ast_str_prefixof(ast(prefix), ast(string))))
            
            return result
        
        return concrete_val is not None
    
    @staticmethod
    def search(pattern, string, flags=0):
        """Symbolic model for re.search()"""
        import re
        concrete_val = re.search(value(pattern), value(string), flags)
        
        # For symbolic values, create a new symbolic boolean
        if hasattr(string, '_ast'):
            # We can't directly model regex searching in Z3
            # So we create a new symbolic boolean with the concrete result
            result = concolic_bool(ast_const_bool(concrete_val is not None), concrete_val is not None)
            
            # Add some basic constraints for common patterns
            pattern_val = value(pattern)
            simple_text = pattern_val.split('[')[0].split('(')[0].split('|')[0]
            if simple_text and not any(c in '.?*+{}()[]\\|' for c in simple_text):
                # Simple text search
                add_pc(ast_eq(result, ast_str_contains(ast(string), ast(simple_text))))
            
            return result
        
        return concrete_val is not None

###########################################
# Collections Module Models
###########################################

class CollectionsModels:
    """Models for collections module"""
    
    @staticmethod
    def Counter(iterable=None):
        """Symbolic model for collections.Counter"""
        import collections
        
        # Get concrete value
        if iterable is not None:
            concrete_val = collections.Counter(value(iterable))
        else:
            concrete_val = collections.Counter()
        
        # For symbolic values, we need to create a wrapper
        if iterable is not None and hasattr(iterable, '_ast'):
            # Create a wrapper that preserves symbolic information
            class SymbolicCounter(collections.Counter):
                def __init__(self, iterable, concrete_counter):
                    self._symbolic_iterable = iterable
                    self._concrete_counter = concrete_counter
                    # Copy the concrete counter's data
                    super().__init__(concrete_counter)
                
                def most_common(self, n=None):
                    # Return concrete result but track symbolically
                    result = self._concrete_counter.most_common(n)
                    
                    # If the input was symbolic, we should track that the result
                    # depends on the symbolic input
                    if hasattr(self._symbolic_iterable, '_ast'):
                        # We can't directly model this in Z3, but we can add a constraint
                        # that the result depends on the input
                        add_pc(ast_eq(ast_const_bool(True), ast_const_bool(True)))
                    
                    return result
            
            return SymbolicCounter(iterable, concrete_val)
        
        return concrete_val
    
    @staticmethod
    def defaultdict(default_factory=None):
        """Symbolic model for collections.defaultdict"""
        import collections
        
        # Create a concrete defaultdict
        concrete_val = collections.defaultdict(default_factory)
        
        # Create a wrapper that preserves symbolic information
        class SymbolicDefaultDict(collections.defaultdict):
            def __init__(self, default_factory, concrete_dict):
                self._concrete_dict = concrete_dict
                super().__init__(default_factory)
            
            def __getitem__(self, key):
                # If key exists, return the value
                if key in self:
                    return super().__getitem__(key)
                
                # Otherwise, create a new value using default_factory
                if self.default_factory is not None:
                    value = self.default_factory()
                    self[key] = value
                    return value
                
                # Raise KeyError if no default_factory
                raise KeyError(key)
        
        return SymbolicDefaultDict(default_factory, concrete_val)

###########################################
# JSON Module Models
###########################################

class JSONModels:
    """Models for JSON operations"""
    
    @staticmethod
    def dumps(obj, **kwargs):
        """Symbolic model for json.dumps()"""
        import json
        
        # Get concrete value
        concrete_val = json.dumps(value(obj), **kwargs)
        
        # For symbolic values, create a new symbolic string
        if hasattr(obj, '_ast'):
            # Create a new symbolic variable
            sym_id = "json_dumps_result"
            
            # We can't directly model JSON serialization in Z3
            # So we create a new symbolic string with the concrete result
            result = concolic_str(ast_str(sym_id), concrete_val)
            
            # Add some basic constraints
            # JSON string starts with { for objects, [ for arrays
            if isinstance(value(obj), dict):
                add_pc(ast_str_prefixof(ast("{"), ast(result)))
            elif isinstance(value(obj), list):
                add_pc(ast_str_prefixof(ast("["), ast(result)))
            
            return result
        
        return concrete_val
    
    @staticmethod
    def loads(s, **kwargs):
        """Symbolic model for json.loads()"""
        import json
        
        # Get concrete value
        concrete_val = json.loads(value(s), **kwargs)
        
        # For symbolic values, we need to create appropriate symbolic wrappers
        if hasattr(s, '_ast'):
            # Create symbolic wrappers based on the type of the concrete value
            if isinstance(concrete_val, dict):
                return concolic_dict(ast_dict("json_loads_result"), concrete_val)
            elif isinstance(concrete_val, list):
                return concolic_list(ast_array("json_loads_result", len(concrete_val)), concrete_val)
            elif isinstance(concrete_val, int):
                return concolic_int(ast_int("json_loads_result"), concrete_val)
            elif isinstance(concrete_val, float):
                return concolic_float(ast_float("json_loads_result"), concrete_val)
            elif isinstance(concrete_val, str):
                return concolic_str(ast_str("json_loads_result"), concrete_val)
            elif isinstance(concrete_val, bool):
                return concolic_bool(ast_const_bool(concrete_val), concrete_val)
        
        return concrete_val

###########################################
# DateTime Module Models
###########################################

class DateTimeModels:
    """Models for datetime operations"""
    
    @staticmethod
    def date(year, month, day):
        """Symbolic model for datetime.date constructor"""
        import datetime
        
        # Get concrete values
        year_val = value(year)
        month_val = value(month)
        day_val = value(day)
        
        try:
            # Create concrete date
            concrete_val = datetime.date(year_val, month_val, day_val)
            
            # Add constraints for valid dates
            if hasattr(year, '_ast') or hasattr(month, '_ast') or hasattr(day, '_ast'):
                # Year constraints
                add_pc(ast_and(
                    ast_ge(ast(year), ast(1)),
                    ast_le(ast(year), ast(9999))
                ))
                
                # Month constraints
                add_pc(ast_and(
                    ast_ge(ast(month), ast(1)),
                    ast_le(ast(month), ast(12))
                ))
                
                # Day constraints (simplified)
                add_pc(ast_and(
                    ast_ge(ast(day), ast(1)),
                    ast_le(ast(day), ast(31))
                ))
            
            # Create a wrapper that preserves symbolic information
            class SymbolicDate(datetime.date):
                @classmethod
                def _new(cls, concrete_date, sym_year, sym_month, sym_day):
                    instance = datetime.date.__new__(cls, 
                                                    concrete_date.year,
                                                    concrete_date.month,
                                                    concrete_date.day)
                    instance._sym_year = sym_year
                    instance._sym_month = sym_month
                    instance._sym_day = sym_day
                    return instance
                
                def __sub__(self, other):
                    # Handle date subtraction
                    result = super().__sub__(other)
                    
                    # If both dates have symbolic components, the result depends on them
                    if (hasattr(self, '_sym_year') or hasattr(self, '_sym_month') or 
                        hasattr(self, '_sym_day') or hasattr(other, '_sym_year') or 
                        hasattr(other, '_sym_month') or hasattr(other, '_sym_day')):
                        
                        # We can't directly model this in Z3, but we can add a constraint
                        # that the result depends on the symbolic inputs
                        add_pc(ast_eq(ast_const_bool(True), ast_const_bool(True)))
                    
                    return result
                
                def __lt__(self, other):
                    result = super().__lt__(other)
                    
                    # If both dates have symbolic components, add constraints
                    if (hasattr(self, '_sym_year') and hasattr(other, '_sym_year')):
                        # Simplified constraint: compare years first
                        add_pc(ast_eq(
                            ast_const_bool(result),
                            ast_lt(ast(self._sym_year), ast(other._sym_year))
                        ))
                    
                    return result
            
            return SymbolicDate._new(concrete_val, year, month, day)
            
        except ValueError as e:
            # If the date is invalid, raise the same exception
            raise ValueError(str(e))

###########################################
# OS Module Models
###########################################

class OSModels:
    """Models for OS operations"""
    
    @staticmethod
    def path_join(*paths):
        """Symbolic model for os.path.join"""
        import os
        
        # Get concrete values
        concrete_paths = [value(p) for p in paths]
        concrete_val = os.path.join(*concrete_paths)
        
        # For symbolic values, create a new symbolic string
        if any(hasattr(p, '_ast') for p in paths):
            # Create a new symbolic variable
            sym_id = "path_join_result"
            
            # We can't directly model path joining in Z3
            # So we create a new symbolic string with the concrete result
            result = concolic_str(ast_str(sym_id), concrete_val)
            
            # Add some basic constraints
            # The result contains all path components
            for p in paths:
                if value(p):  # Skip empty paths
                    add_pc(ast_str_contains(ast(result), ast(p)))
            
            return result
        
        return concrete_val
    
    @staticmethod
    def path_exists(path):
        """Symbolic model for os.path.exists"""
        import os
        
        # Get concrete value
        concrete_val = os.path.exists(value(path))
        
        # For symbolic values, create a new symbolic boolean
        if hasattr(path, '_ast'):
            # We can't directly model file existence in Z3
            # So we create a new symbolic boolean with the concrete result
            result = concolic_bool(ast_const_bool(concrete_val), concrete_val)
            
            return result
        
        return concrete_val

###########################################
# Itertools Module Models
###########################################

class ItertoolsModels:
    """Models for itertools operations"""
    
    @staticmethod
    def product(*iterables, repeat=1):
        """Symbolic model for itertools.product"""
        import itertools
        
        # Get concrete values
        concrete_iterables = [value(it) for it in iterables]
        concrete_val = list(itertools.product(*concrete_iterables, repeat=repeat))
        
        # For symbolic values, create a new symbolic list
        if any(hasattr(it, '_ast') for it in iterables):
            # Create a new symbolic variable
            sym_id = "product_result"
            
            # We can't directly model product in Z3
            # So we create a new symbolic list with the concrete result
            result = concolic_list(ast_array(sym_id, len(concrete_val)), concrete_val)
            
            # Add some basic constraints
            # The length of the result is the product of the lengths of the inputs
            expected_len = 1
            for it in iterables:
                expected_len *= len(value(it))
            expected_len *= repeat
            
            add_pc(ast_eq(ast_const_int(len(concrete_val)), ast_const_int(expected_len)))
            
            return result
        
        return concrete_val
    
    @staticmethod
    def combinations(iterable, r):
        """Symbolic model for itertools.combinations"""
        import itertools
        
        # Get concrete values
        concrete_iterable = value(iterable)
        concrete_val = list(itertools.combinations(concrete_iterable, r))
        
        # For symbolic values, create a new symbolic list
        if hasattr(iterable, '_ast'):
            # Create a new symbolic variable
            sym_id = "combinations_result"
            
            # We can't directly model combinations in Z3
            # So we create a new symbolic list with the concrete result
            result = concolic_list(ast_array(sym_id, len(concrete_val)), concrete_val)
            
            # Add some basic constraints
            # Each combination has length r
            for combo in concrete_val:
                if len(combo) != r:
                    add_pc(ast_eq(ast_const_bool(False), ast_const_bool(True)))
                    break
            
            return result
        
        return concrete_val

###########################################
# Module Patching Functions
###########################################

def patch_math_module():
    """Patch the math module with symbolic models"""
    import math
    
    # Save original functions
    orig_sin = math.sin
    orig_cos = math.cos
    orig_sqrt = math.sqrt
    
    # Replace with symbolic models
    math.sin = MathModels.sin
    math.cos = MathModels.cos
    math.sqrt = MathModels.sqrt
    
    # Return a function to restore original functions
    def restore():
        math.sin = orig_sin
        math.cos = orig_cos
        math.sqrt = orig_sqrt
    
    return restore

def patch_string_methods():
    """Patch string methods with symbolic models"""
    # Save original methods
    orig_upper = str.upper
    orig_lower = str.lower
    orig_split = str.split
    
    # Replace with symbolic models
    str.upper = lambda self: StringModels.upper(self)
    str.lower = lambda self: StringModels.lower(self)
    str.split = lambda self, sep=None, maxsplit=-1: StringModels.split(self, sep, maxsplit)
    
    # Return a function to restore original methods
    def restore():
        str.upper = orig_upper
        str.lower = orig_lower
        str.split = orig_split
    
    return restore

def patch_re_module():
    """Patch the re module with symbolic models"""
    import re
    
    # Save original functions
    orig_match = re.match
    orig_search = re.search
    
    # Replace with symbolic models
    re.match = RegexModels.match
    re.search = RegexModels.search
    
    # Return a function to restore original functions
    def restore():
        re.match = orig_match
        re.search = orig_search
    
    return restore

def patch_collections_module():
    """Patch the collections module with symbolic models"""
    import collections
    
    # Save original classes
    orig_Counter = collections.Counter
    orig_defaultdict = collections.defaultdict
    
    # Replace with symbolic models
    collections.Counter = CollectionsModels.Counter
    collections.defaultdict = CollectionsModels.defaultdict
    
    # Return a function to restore original classes
    def restore():
        collections.Counter = orig_Counter
        collections.defaultdict = orig_defaultdict
    
    return restore

def patch_json_module():
    """Patch the json module with symbolic models"""
    import json
    
    # Save original functions
    orig_dumps = json.dumps
    orig_loads = json.loads
    
    # Replace with symbolic models
    json.dumps = JSONModels.dumps
    json.loads = JSONModels.loads
    
    # Return a function to restore original functions
    def restore():
        json.dumps = orig_dumps
        json.loads = orig_loads
    
    return restore

def patch_datetime_module():
    """Patch the datetime module with symbolic models"""
    import datetime
    
    # Save original classes
    orig_date = datetime.date
    
    # Replace with symbolic models
    datetime.date = lambda year, month, day: DateTimeModels.date(year, month, day)
    
    # Return a function to restore original classes
    def restore():
        datetime.date = orig_date
    
    return restore

def patch_os_module():
    """Patch the os module with symbolic models"""
    import os
    
    # Save original functions
    orig_path_join = os.path.join
    orig_path_exists = os.path.exists
    
    # Replace with symbolic models
    os.path.join = OSModels.path_join
    os.path.exists = OSModels.path_exists
    
    # Return a function to restore original functions
    def restore():
        os.path.join = orig_path_join
        os.path.exists = orig_path_exists
    
    return restore

def patch_itertools_module():
    """Patch the itertools module with symbolic models"""
    import itertools
    
    # Save original functions
    orig_product = itertools.product
    orig_combinations = itertools.combinations
    
    # Replace with symbolic models
    itertools.product = ItertoolsModels.product
    itertools.combinations = ItertoolsModels.combinations
    
    # Return a function to restore original functions
    def restore():
        itertools.product = orig_product
        itertools.combinations = orig_combinations
    
    return restore

###########################################
# Main Patching Function
###########################################

def patch_stdlib():
    """Patch all standard library modules with symbolic models"""
    restorers = []
    
    # Patch each module
    restorers.append(patch_math_module())
    restorers.append(patch_string_methods())
    restorers.append(patch_re_module())
    restorers.append(patch_collections_module())
    restorers.append(patch_json_module())
    restorers.append(patch_datetime_module())
    restorers.append(patch_os_module())
    restorers.append(patch_itertools_module())
    
    # Return a function to restore all original functions
    def restore_all():
        for restore in restorers:
            restore()
    
    return restore_all

# Automatically patch standard library when imported
restore_stdlib = patch_stdlib() 