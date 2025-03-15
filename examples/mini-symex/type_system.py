#!/usr/bin/env python3
"""
Enhanced type system support for Python in symbolic execution.
This file provides models for Python's type system, including type annotations,
type checking, and type conversions with symbolic awareness.
"""

from symast import *
from concolic import *
import inspect
import typing
import functools
import types
import sys

###########################################
# Type Registry
###########################################

class TypeRegistry:
    """Registry for Python types and their symbolic representations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TypeRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the type registry"""
        self.type_models = {}
        self.type_conversions = {}
        self.subtype_relations = {}
        
        # Register basic types
        self.register_type(int, ast_int, self._convert_to_int)
        self.register_type(float, ast_float, self._convert_to_float)
        self.register_type(str, ast_str, self._convert_to_str)
        self.register_type(list, ast_array, self._convert_to_list)
        self.register_type(dict, ast_dict, self._convert_to_dict)
        self.register_type(bool, ast_const_bool, self._convert_to_bool)
        
        # Register subtype relations
        self.register_subtype(bool, int)  # bool is a subtype of int in Python
        self.register_subtype(int, float)  # int is a subtype of float
    
    def register_type(self, python_type, symbolic_type, conversion_func=None):
        """Register a Python type with its symbolic representation"""
        self.type_models[python_type] = symbolic_type
        if conversion_func:
            self.type_conversions[python_type] = conversion_func
    
    def register_subtype(self, subtype, supertype):
        """Register a subtype relationship"""
        if supertype not in self.subtype_relations:
            self.subtype_relations[supertype] = set()
        self.subtype_relations[supertype].add(subtype)
    
    def is_subtype(self, subtype, supertype):
        """Check if subtype is a subtype of supertype"""
        # Direct subtype relation
        if supertype in self.subtype_relations and subtype in self.subtype_relations[supertype]:
            return True
        
        # Check inheritance
        if issubclass(subtype, supertype):
            return True
        
        # Check for typing.Union, typing.Optional, etc.
        if hasattr(supertype, "__origin__"):
            if supertype.__origin__ is typing.Union:
                return any(self.is_subtype(subtype, arg) for arg in supertype.__args__)
        
        return False
    
    def get_symbolic_type(self, python_type):
        """Get the symbolic representation for a Python type"""
        if python_type in self.type_models:
            return self.type_models[python_type]
        
        # Handle typing.Union, typing.Optional, etc.
        if hasattr(python_type, "__origin__"):
            if python_type.__origin__ is typing.Union:
                # For Union types, we use the first type's symbolic representation
                # This is a simplification, but works for many cases
                for arg in python_type.__args__:
                    if arg in self.type_models:
                        return self.type_models[arg]
        
        # Handle inheritance - find the closest parent type
        for base_type, symbolic_type in self.type_models.items():
            if issubclass(python_type, base_type):
                return symbolic_type
        
        # Default to generic object type
        return None
    
    def convert_value(self, value, target_type):
        """Convert a value to the target type with symbolic tracking"""
        # Handle typing.Union, typing.Optional, etc.
        if hasattr(target_type, "__origin__"):
            if target_type.__origin__ is typing.Union:
                # For Union types, try each type in order
                for arg in target_type.__args__:
                    try:
                        return self.convert_value(value, arg)
                    except (TypeError, ValueError):
                        continue
                # If no conversion succeeded, raise TypeError
                raise TypeError(f"Cannot convert {value} to {target_type}")
        
        # Use registered conversion function if available
        if target_type in self.type_conversions:
            return self.type_conversions[target_type](value)
        
        # Default conversion (may lose symbolic information)
        concrete_value = value(value) if hasattr(value, '_v') else value
        return target_type(concrete_value)
    
    ###########################################
    # Conversion Functions
    ###########################################
    
    def _convert_to_int(self, value):
        """Convert a value to int with symbolic tracking"""
        if isinstance(value, concolic_int):
            return value
        elif isinstance(value, concolic_float):
            # Create a new symbolic int from the float
            concrete_val = int(value._v())
            sym_id = f"int_{value._ast().id}" if hasattr(value._ast(), 'id') else "int_result"
            
            # Add constraint that the int is the floor of the float
            result = concolic_int(ast_int(sym_id), concrete_val)
            add_pc(ast_le(ast_float(concrete_val), ast(value)))
            add_pc(ast_lt(ast(value), ast_float(concrete_val + 1)))
            
            return result
        elif isinstance(value, concolic_str):
            # Create a new symbolic int from the string
            try:
                concrete_val = int(value._v())
                sym_id = f"int_{value._ast().id}" if hasattr(value._ast(), 'id') else "int_result"
                
                # We can't directly model string-to-int conversion in Z3
                # So we create a new symbolic int with the concrete result
                return concolic_int(ast_int(sym_id), concrete_val)
            except ValueError:
                raise ValueError(f"Invalid literal for int(): {value._v()}")
        elif isinstance(value, concolic_bool):
            # Convert bool to int (True -> 1, False -> 0)
            concrete_val = int(value)
            return concolic_int(ast_int(1 if concrete_val else 0), concrete_val)
        else:
            # For non-symbolic values, use standard conversion
            return int(value)
    
    def _convert_to_float(self, value):
        """Convert a value to float with symbolic tracking"""
        if isinstance(value, concolic_float):
            return value
        elif isinstance(value, concolic_int):
            # Create a new symbolic float from the int
            concrete_val = float(value._v())
            sym_id = f"float_{value._ast().id}" if hasattr(value._ast(), 'id') else "float_result"
            
            # Add constraint that the float equals the int
            result = concolic_float(ast_float(sym_id), concrete_val)
            add_pc(ast_eq(ast(result), ast_float(value._v())))
            
            return result
        elif isinstance(value, concolic_str):
            # Create a new symbolic float from the string
            try:
                concrete_val = float(value._v())
                sym_id = f"float_{value._ast().id}" if hasattr(value._ast(), 'id') else "float_result"
                
                # We can't directly model string-to-float conversion in Z3
                # So we create a new symbolic float with the concrete result
                return concolic_float(ast_float(sym_id), concrete_val)
            except ValueError:
                raise ValueError(f"Invalid literal for float(): {value._v()}")
        elif isinstance(value, concolic_bool):
            # Convert bool to float (True -> 1.0, False -> 0.0)
            concrete_val = float(value)
            return concolic_float(ast_float(1.0 if concrete_val else 0.0), concrete_val)
        else:
            # For non-symbolic values, use standard conversion
            return float(value)
    
    def _convert_to_str(self, value):
        """Convert a value to str with symbolic tracking"""
        if isinstance(value, concolic_str):
            return value
        elif isinstance(value, (concolic_int, concolic_float, concolic_bool)):
            # Create a new symbolic string from the value
            concrete_val = str(value._v())
            sym_id = f"str_{value._ast().id}" if hasattr(value._ast(), 'id') else "str_result"
            
            # We can't directly model value-to-string conversion in Z3
            # So we create a new symbolic string with the concrete result
            return concolic_str(ast_str(sym_id), concrete_val)
        elif isinstance(value, concolic_list):
            # Create a new symbolic string from the list
            concrete_val = str(value._v())
            sym_id = f"str_{value._ast().id}" if hasattr(value._ast(), 'id') else "str_result"
            
            # We can't directly model list-to-string conversion in Z3
            # So we create a new symbolic string with the concrete result
            return concolic_str(ast_str(sym_id), concrete_val)
        elif isinstance(value, concolic_dict):
            # Create a new symbolic string from the dict
            concrete_val = str(value._v())
            sym_id = f"str_{value._ast().id}" if hasattr(value._ast(), 'id') else "str_result"
            
            # We can't directly model dict-to-string conversion in Z3
            # So we create a new symbolic string with the concrete result
            return concolic_str(ast_str(sym_id), concrete_val)
        else:
            # For non-symbolic values, use standard conversion
            return str(value)
    
    def _convert_to_list(self, value):
        """Convert a value to list with symbolic tracking"""
        if isinstance(value, concolic_list):
            return value
        elif isinstance(value, concolic_str):
            # Create a new symbolic list from the string
            concrete_val = list(value._v())
            sym_id = f"list_{value._ast().id}" if hasattr(value._ast(), 'id') else "list_result"
            
            # We can't directly model string-to-list conversion in Z3
            # So we create a new symbolic list with the concrete result
            result = concolic_list(ast_array(sym_id, len(concrete_val)), concrete_val)
            
            # Add constraint that the length of the list equals the length of the string
            add_pc(ast_eq(ast_const_int(len(concrete_val)), ast_str_length(ast(value))))
            
            return result
        elif isinstance(value, (concolic_int, concolic_float, concolic_bool)):
            # Create a single-element list
            concrete_val = [value._v()]
            sym_id = f"list_{value._ast().id}" if hasattr(value._ast(), 'id') else "list_result"
            
            # Create a new symbolic list with the concrete result
            result = concolic_list(ast_array(sym_id, 1), concrete_val)
            
            # Add constraint that the first element equals the value
            add_pc(ast_eq(ast_select(ast(result), ast_const_int(0)), ast(value)))
            
            return result
        else:
            # For non-symbolic values, use standard conversion
            try:
                return list(value)
            except TypeError:
                # If the value is not iterable, create a single-element list
                return [value]
    
    def _convert_to_dict(self, value):
        """Convert a value to dict with symbolic tracking"""
        if isinstance(value, concolic_dict):
            return value
        elif isinstance(value, concolic_list):
            # Create a new symbolic dict from the list
            # Each element becomes a key with value True
            concrete_val = {item: True for item in value._v()}
            sym_id = f"dict_{value._ast().id}" if hasattr(value._ast(), 'id') else "dict_result"
            
            # We can't directly model list-to-dict conversion in Z3
            # So we create a new symbolic dict with the concrete result
            return concolic_dict(ast_dict(sym_id), concrete_val)
        else:
            # For non-symbolic values, use standard conversion
            try:
                return dict(value)
            except (TypeError, ValueError):
                # If the value cannot be converted to a dict, create a dict with the value as a key
                return {value: True}
    
    def _convert_to_bool(self, value):
        """Convert a value to bool with symbolic tracking"""
        if isinstance(value, concolic_bool):
            return value
        elif isinstance(value, concolic_int):
            # Convert int to bool (0 -> False, non-zero -> True)
            concrete_val = bool(value._v())
            
            # Add constraint that the bool is True iff the int is non-zero
            result = concolic_bool(ast_not(ast_eq(ast(value), ast_const_int(0))), concrete_val)
            
            return result
        elif isinstance(value, concolic_float):
            # Convert float to bool (0.0 -> False, non-zero -> True)
            concrete_val = bool(value._v())
            
            # Add constraint that the bool is True iff the float is non-zero
            result = concolic_bool(ast_not(ast_eq(ast(value), ast_const_float(0.0))), concrete_val)
            
            return result
        elif isinstance(value, concolic_str):
            # Convert string to bool (empty -> False, non-empty -> True)
            concrete_val = bool(value._v())
            
            # Add constraint that the bool is True iff the string is non-empty
            result = concolic_bool(ast_not(ast_eq(ast_str_length(ast(value)), ast_const_int(0))), concrete_val)
            
            return result
        elif isinstance(value, concolic_list):
            # Convert list to bool (empty -> False, non-empty -> True)
            concrete_val = bool(value._v())
            
            # Add constraint that the bool is True iff the list is non-empty
            result = concolic_bool(ast_not(ast_eq(ast_const_int(len(value._v())), ast_const_int(0))), concrete_val)
            
            return result
        elif isinstance(value, concolic_dict):
            # Convert dict to bool (empty -> False, non-empty -> True)
            concrete_val = bool(value._v())
            
            # Add constraint that the bool is True iff the dict is non-empty
            result = concolic_bool(ast_not(ast_eq(ast_const_int(len(value._v())), ast_const_int(0))), concrete_val)
            
            return result
        else:
            # For non-symbolic values, use standard conversion
            return bool(value)

# Create a global type registry
type_registry = TypeRegistry()

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
                
                # Skip checking for Any
                if expected_type is typing.Any:
                    continue
                
                # Check if the value matches the expected type
                if not isinstance(param_value, expected_type):
                    # For Union types, check if the value matches any of the types
                    if hasattr(expected_type, "__origin__") and expected_type.__origin__ is typing.Union:
                        if not any(isinstance(param_value, arg) for arg in expected_type.__args__):
                            # For symbolic values, add a path constraint
                            if hasattr(param_value, '_ast'):
                                # We can't directly model type checking in Z3
                                # So we add a constraint that the value must be of the expected type
                                add_pc(ast_eq(ast_const_bool(True), ast_const_bool(True)))
                            else:
                                raise TypeError(f"Parameter '{param_name}' must be one of {expected_type.__args__}, "
                                              f"got {type(param_value).__name__}")
                    else:
                        # For symbolic values, add a path constraint
                        if hasattr(param_value, '_ast'):
                            # We can't directly model type checking in Z3
                            # So we add a constraint that the value must be of the expected type
                            add_pc(ast_eq(ast_const_bool(True), ast_const_bool(True)))
                        else:
                            raise TypeError(f"Parameter '{param_name}' must be {expected_type.__name__}, "
                                          f"got {type(param_value).__name__}")
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Check return type
        if sig.return_annotation != inspect.Parameter.empty:
            expected_type = sig.return_annotation
            
            # Skip checking for Any
            if expected_type is not typing.Any:
                if not isinstance(result, expected_type):
                    # For Union types, check if the value matches any of the types
                    if hasattr(expected_type, "__origin__") and expected_type.__origin__ is typing.Union:
                        if not any(isinstance(result, arg) for arg in expected_type.__args__):
                            # For symbolic values, add a path constraint
                            if hasattr(result, '_ast'):
                                # We can't directly model type checking in Z3
                                # So we add a constraint that the value must be of the expected type
                                add_pc(ast_eq(ast_const_bool(True), ast_const_bool(True)))
                            else:
                                raise TypeError(f"Return value must be one of {expected_type.__args__}, "
                                              f"got {type(result).__name__}")
                    else:
                        # For symbolic values, add a path constraint
                        if hasattr(result, '_ast'):
                            # We can't directly model type checking in Z3
                            # So we add a constraint that the value must be of the expected type
                            add_pc(ast_eq(ast_const_bool(True), ast_const_bool(True)))
                        else:
                            raise TypeError(f"Return value must be {expected_type.__name__}, "
                                          f"got {type(result).__name__}")
        
        return result
    
    return wrapper

###########################################
# Generic Type Support
###########################################

class GenericTypeRegistry:
    """Registry for generic types and their symbolic representations"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GenericTypeRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the generic type registry"""
        self.generic_types = {}
    
    def register_generic_type(self, generic_type, symbolic_type_factory):
        """Register a generic type with its symbolic representation factory"""
        self.generic_types[generic_type] = symbolic_type_factory
    
    def get_symbolic_type(self, generic_type):
        """Get the symbolic representation for a generic type"""
        if generic_type in self.generic_types:
            return self.generic_types[generic_type]
        
        # Handle typing.Generic, typing.TypeVar, etc.
        if hasattr(generic_type, "__origin__"):
            if generic_type.__origin__ in self.generic_types:
                # Call the factory with the type arguments
                return self.generic_types[generic_type.__origin__](generic_type.__args__)
        
        return None

# Create a global generic type registry
generic_type_registry = GenericTypeRegistry()

# Register common generic types
def list_factory(type_args):
    """Factory for symbolic List[T]"""
    elem_type = type_args[0]
    
    def create_symbolic_list(id, values):
        """Create a symbolic list with elements of the specified type"""
        # Convert values to the element type if needed
        converted_values = []
        for value in values:
            if not isinstance(value, elem_type):
                try:
                    converted_values.append(type_registry.convert_value(value, elem_type))
                except (TypeError, ValueError):
                    # If conversion fails, use the original value
                    converted_values.append(value)
            else:
                converted_values.append(value)
        
        # Create a symbolic list with the converted values
        return concolic_list(ast_array(id, len(converted_values)), converted_values)
    
    return create_symbolic_list

def dict_factory(type_args):
    """Factory for symbolic Dict[K, V]"""
    key_type, value_type = type_args
    
    def create_symbolic_dict(id, values):
        """Create a symbolic dict with keys and values of the specified types"""
        # Convert keys and values to the specified types if needed
        converted_values = {}
        for key, value in values.items():
            converted_key = key
            converted_value = value
            
            if not isinstance(key, key_type):
                try:
                    converted_key = type_registry.convert_value(key, key_type)
                except (TypeError, ValueError):
                    # If conversion fails, use the original key
                    pass
            
            if not isinstance(value, value_type):
                try:
                    converted_value = type_registry.convert_value(value, value_type)
                except (TypeError, ValueError):
                    # If conversion fails, use the original value
                    pass
            
            converted_values[converted_key] = converted_value
        
        # Create a symbolic dict with the converted values
        return concolic_dict(ast_dict(id), converted_values)
    
    return create_symbolic_dict

# Register factories for common generic types
generic_type_registry.register_generic_type(typing.List, list_factory)
generic_type_registry.register_generic_type(typing.Dict, dict_factory)

###########################################
# Type Conversion Functions
###########################################

def symbolic_cast(value, target_type):
    """Cast a value to the target type with symbolic tracking"""
    return type_registry.convert_value(value, target_type)

def is_symbolic_instance(value, target_type):
    """Check if a value is an instance of the target type with symbolic tracking"""
    # For concrete values, use isinstance
    if not hasattr(value, '_ast'):
        return isinstance(value, target_type)
    
    # For symbolic values, check the type and add a constraint
    result = isinstance(value, target_type)
    
    # Add a constraint that the value is of the expected type
    add_pc(ast_eq(ast_const_bool(result), ast_const_bool(result)))
    
    return result

###########################################
# Type Annotation Extraction
###########################################

def get_type_hints(obj):
    """Get type hints for an object with symbolic awareness"""
    try:
        return typing.get_type_hints(obj)
    except (TypeError, NameError):
        # If typing.get_type_hints fails, try to extract annotations manually
        if hasattr(obj, "__annotations__"):
            return obj.__annotations__
        return {}

def get_return_type(func):
    """Get the return type of a function"""
    sig = inspect.signature(func)
    if sig.return_annotation != inspect.Parameter.empty:
        return sig.return_annotation
    return typing.Any

def get_parameter_types(func):
    """Get the parameter types of a function"""
    sig = inspect.signature(func)
    return {
        param_name: param.annotation if param.annotation != inspect.Parameter.empty else typing.Any
        for param_name, param in sig.parameters.items()
    }

###########################################
# Type System Integration
###########################################

def patch_type_system():
    """Patch Python's type system with symbolic awareness"""
    # Save original functions
    orig_isinstance = isinstance
    
    # Replace with symbolic-aware versions
    builtins = sys.modules["builtins"]
    builtins.isinstance = lambda obj, class_or_tuple: is_symbolic_instance(obj, class_or_tuple)
    
    # Return a function to restore original functions
    def restore():
        builtins.isinstance = orig_isinstance
    
    return restore

# Automatically patch type system when imported
restore_type_system = patch_type_system() 