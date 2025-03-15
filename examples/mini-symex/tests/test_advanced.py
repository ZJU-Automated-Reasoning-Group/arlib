#!/usr/bin/env python3
"""
Test cases for Python's advanced language features using the extended mini-symex framework.
This file tests more complex language constructs and patterns.
"""


import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
import functools
import contextlib
import types
import inspect
# Add parent directory to path to allow importing modules
###########################################
# 1. Decorators and Metaprogramming
###########################################

def test_decorators():
    """Test function decorators with symbolic values"""
    print("\n=== Testing Decorators ===")
    
    # Define decorators
    def log_calls(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            print(f"{func.__name__} returned: {result}")
            return result
        return wrapper
    
    def multiply_result(factor):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return result * factor
            return wrapper
        return decorator
    
    # Apply decorators
    @log_calls
    @multiply_result(2)
    def add(a, b):
        return a + b
    
    # Test with symbolic values
    x = mk_int("x")
    y = mk_int("y")
    
    result = add(x, y)
    
    if result > 20:
        print(f"Decorated add(x, y) > 20: x={x}, y={y}, result={result}")

def test_class_decorators():
    """Test class decorators with symbolic values"""
    print("\n=== Testing Class Decorators ===")
    
    # Define a class decorator
    def add_methods(cls):
        def new_method(self, x):
            return self.value + x
        
        cls.add = new_method
        return cls
    
    # Apply the decorator
    @add_methods
    class MyClass:
        def __init__(self, value):
            self.value = value
    
    # Test with symbolic values
    x = mk_int("x")
    
    # Create an instance
    obj = MyClass(x)
    
    # Call the dynamically added method
    y = mk_int("y")
    result = obj.add(y)
    
    if result > 15:
        print(f"obj.add(y) > 15: x={x}, y={y}, result={result}")

def test_metaclasses():
    """Test metaclasses with symbolic values"""
    print("\n=== Testing Metaclasses ===")
    
    # Define a tracing decorator
    def trace(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__}")
            result = func(*args, **kwargs)
            print(f"{func.__name__} returned {result}")
            return result
        return wrapper
    
    # Define a metaclass
    class TracingMeta(type):
        def __new__(mcs, name, bases, namespace):
            # Wrap all methods with a tracing decorator
            for key, value in namespace.items():
                if callable(value) and not key.startswith('__'):
                    namespace[key] = trace(value)
            
            return super().__new__(mcs, name, bases, namespace)
    
    # Create a class with the metaclass
    class MyClass(metaclass=TracingMeta):
        def __init__(self, value):
            self.value = value
        
        def double(self):
            return self.value * 2
    
    # Test with symbolic values
    x = mk_int("x")
    
    # Create an instance
    obj = MyClass(x)
    
    # Call a method
    result = obj.double()
    
    if result > 10:
        print(f"obj.double() > 10: x={x}, result={result}")

###########################################
# 2. Context Managers
###########################################

def test_context_managers():
    """Test context managers with symbolic values"""
    print("\n=== Testing Context Managers ===")
    
    # Define a context manager
    @contextlib.contextmanager
    def symbolic_context(value):
        print(f"Entering context with value: {value}")
        try:
            yield value * 2
        finally:
            print(f"Exiting context with value: {value}")
    
    # Test with symbolic values
    x = mk_int("x")
    
    # Use the context manager
    with symbolic_context(x) as doubled:
        if doubled > 10:
            print(f"doubled > 10: x={x}, doubled={doubled}")
        
        y = mk_int("y")
        result = doubled + y
        
        if result > 15:
            print(f"result > 15: doubled={doubled}, y={y}, result={result}")

def test_custom_context_manager_class():
    """Test custom context manager class with symbolic values"""
    print("\n=== Testing Custom Context Manager Class ===")
    
    # Define a context manager class
    class SymbolicContext:
        def __init__(self, value):
            self.value = value
        
        def __enter__(self):
            print(f"Entering context with value: {self.value}")
            return self.value * 2
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            print(f"Exiting context with value: {self.value}")
            # Return True to suppress exceptions
            return True
    
    # Test with symbolic values
    x = mk_int("x")
    
    # Use the context manager
    with SymbolicContext(x) as doubled:
        if doubled > 10:
            print(f"doubled > 10: x={x}, doubled={doubled}")
        
        # This will raise an exception, but it will be suppressed
        if doubled > 15:
            raise ValueError("This exception will be suppressed")
            
        print("This will be executed even after the exception")

###########################################
# 3. Generators and Coroutines
###########################################

def test_generators():
    """Test generators with symbolic values"""
    print("\n=== Testing Generators ===")
    
    # Define a generator function
    def count_up_to(limit):
        i = 0
        while i < limit:
            yield i
            i += 1
    
    # Test with symbolic values
    x = mk_int("x")
    
    # Ensure x is within a reasonable range
    if 0 <= x <= 10:
        # Use the generator
        gen = count_up_to(x)
        
        # Convert to list to use it
        values = list(gen)
        
        if len(values) > 5:
            print(f"Generated more than 5 values: x={x}, values={values}")
        
        # Sum the generated values
        total = sum(values)
        
        if total > 10:
            print(f"Sum of generated values > 10: x={x}, total={total}")

def test_generator_expressions():
    """Test generator expressions with symbolic values"""
    print("\n=== Testing Generator Expressions ===")
    
    # Test with symbolic values
    x = mk_int("x")
    y = mk_int("y")
    
    # Create a generator expression
    gen = (i * x for i in range(y) if i % 2 == 0)
    
    # Convert to list to use it
    values = list(gen)
    
    if len(values) > 2:
        print(f"Generated more than 2 values: x={x}, y={y}, values={values}")
    
    # Sum the generated values
    total = sum(values)
    
    if total > 15:
        print(f"Sum of generated values > 15: x={x}, y={y}, total={total}")

###########################################
# 4. Dynamic Code Execution
###########################################

def test_dynamic_code():
    """Test dynamic code execution with symbolic values"""
    print("\n=== Testing Dynamic Code Execution ===")
    
    # Test with symbolic values
    x = mk_int("x")
    
    # Create a code string
    code_str = f"result = {x} * 2 + 5"
    
    # Create a namespace
    namespace = {"x": x}
    
    # Execute the code
    exec(code_str, globals(), namespace)
    
    # Get the result
    result = namespace["result"]
    
    if result > 15:
        print(f"Dynamic code result > 15: x={x}, result={result}")

def test_dynamic_function_creation():
    """Test dynamic function creation with symbolic values"""
    print("\n=== Testing Dynamic Function Creation ===")
    
    # Test with symbolic values
    x = mk_int("x")
    
    # Create a function dynamically
    func_code = f"""
def dynamic_func(y):
    return {x} * y + 10
"""
    
    # Create a namespace
    namespace = {"x": x}
    
    # Execute the code to define the function
    exec(func_code, globals(), namespace)
    
    # Get the function
    dynamic_func = namespace["dynamic_func"]
    
    # Call the function
    y = mk_int("y")
    result = dynamic_func(y)
    
    if result > 20:
        print(f"Dynamic function result > 20: x={x}, y={y}, result={result}")

###########################################
# 5. Reflection and Introspection
###########################################

def test_reflection():
    """Test reflection and introspection with symbolic values"""
    print("\n=== Testing Reflection and Introspection ===")
    
    # Define a class with symbolic attributes
    class ReflectiveClass:
        def __init__(self, a, b):
            self.a = a
            self.b = b
        
        def method1(self):
            return self.a + self.b
        
        def method2(self):
            return self.a * self.b
    
    # Test with symbolic values
    x = mk_int("x")
    y = mk_int("y")
    
    # Create an instance
    obj = ReflectiveClass(x, y)
    
    # Get attributes using getattr
    a_value = getattr(obj, "a")
    b_value = getattr(obj, "b")
    
    if a_value + b_value > 10:
        print(f"a + b > 10: a={a_value}, b={b_value}")
    
    # Get methods using dir and inspect
    methods = [name for name in dir(obj) if callable(getattr(obj, name)) and not name.startswith('__')]
    print(f"Methods of ReflectiveClass: {methods}")
    
    # Call a method dynamically
    method_name = "method1" if x > y else "method2"
    method = getattr(obj, method_name)
    result = method()
    
    print(f"Dynamically called {method_name}: result={result}")

def test_type_annotations():
    """Test type annotations with symbolic values"""
    print("\n=== Testing Type Annotations ===")
    
    # Define a function with type annotations
    def annotated_func(a: int, b: int) -> int:
        return a + b
    
    # Test with symbolic values
    x = mk_int("x")
    y = mk_int("y")
    
    # Call the function
    result = annotated_func(x, y)
    
    if result > 10:
        print(f"annotated_func(x, y) > 10: x={x}, y={y}, result={result}")
    
    # Inspect the annotations
    annotations = annotated_func.__annotations__
    print(f"Annotations of annotated_func: {annotations}")
    
    # Check if the types match
    if isinstance(x, annotations['a']) and isinstance(y, annotations['b']):
        print(f"Types match the annotations")
    else:
        print(f"Types don't match the annotations")

###########################################
# 6. Advanced Class Features
###########################################

def test_descriptors():
    """Test descriptors with symbolic values"""
    print("\n=== Testing Descriptors ===")
    
    # Define a descriptor
    class SymbolicDescriptor:
        def __init__(self, name):
            self.name = name
            self.value = None
        
        def __get__(self, instance, owner):
            print(f"Getting {self.name}: {self.value}")
            return self.value
        
        def __set__(self, instance, value):
            print(f"Setting {self.name} to {value}")
            self.value = value
    
    # Define a class with descriptors
    class DescriptorClass:
        x = SymbolicDescriptor("x")
        y = SymbolicDescriptor("y")
        
        def __init__(self, x_val, y_val):
            self.x = x_val
            self.y = y_val
        
        def sum(self):
            return self.x + self.y
    
    # Test with symbolic values
    x = mk_int("x")
    y = mk_int("y")
    
    # Create an instance
    obj = DescriptorClass(x, y)
    
    # Access the attributes
    x_value = obj.x
    y_value = obj.y
    
    if x_value + y_value > 10:
        print(f"x + y > 10: x={x_value}, y={y_value}")
    
    # Call a method
    result = obj.sum()
    
    if result > 15:
        print(f"obj.sum() > 15: result={result}")

def test_slots():
    """Test __slots__ with symbolic values"""
    print("\n=== Testing __slots__ ===")
    
    # Define a class with __slots__
    class SlottedClass:
        __slots__ = ['x', 'y']
        
        def __init__(self, x, y):
            self.x = x
            self.y = y
        
        def sum(self):
            return self.x + self.y
    
    # Test with symbolic values
    x = mk_int("x")
    y = mk_int("y")
    
    # Create an instance
    obj = SlottedClass(x, y)
    
    # Access the attributes
    x_value = obj.x
    y_value = obj.y
    
    if x_value + y_value > 10:
        print(f"x + y > 10: x={x_value}, y={y_value}")
    
    # Try to add a new attribute (should raise an AttributeError)
    try:
        obj.z = x + y
        print("Added z attribute (unexpected)")
    except AttributeError:
        print("Cannot add z attribute (expected)")

def test_property_decorators():
    """Test property decorators with symbolic values"""
    print("\n=== Testing Property Decorators ===")
    
    # Define a class with properties
    class PropertyClass:
        def __init__(self, x, y):
            self._x = x
            self._y = y
        
        @property
        def x(self):
            print(f"Getting x: {self._x}")
            return self._x
        
        @x.setter
        def x(self, value):
            print(f"Setting x to {value}")
            self._x = value
        
        @property
        def y(self):
            print(f"Getting y: {self._y}")
            return self._y
        
        @y.setter
        def y(self, value):
            print(f"Setting y to {value}")
            self._y = value
        
        @property
        def sum(self):
            return self._x + self._y
    
    # Test with symbolic values
    x = mk_int("x")
    y = mk_int("y")
    
    # Create an instance
    obj = PropertyClass(x, y)
    
    # Access the properties
    x_value = obj.x
    y_value = obj.y
    
    if x_value + y_value > 10:
        print(f"x + y > 10: x={x_value}, y={y_value}")
    
    # Set new values
    obj.x = x * 2
    obj.y = y * 2
    
    # Get the sum property
    sum_value = obj.sum
    
    if sum_value > 20:
        print(f"obj.sum > 20: sum={sum_value}")

###########################################
# Main Test Runner
###########################################

if __name__ == "__main__":
    print("=== Running Advanced Language Features Tests ===")
    
    # Decorators and Metaprogramming
    concolic(test_decorators, debug=True, exit_on_err=False)
    concolic(test_class_decorators, debug=True, exit_on_err=False)
    concolic(test_metaclasses, debug=True, exit_on_err=False)
    
    # Context Managers
    concolic(test_context_managers, debug=True, exit_on_err=False)
    concolic(test_custom_context_manager_class, debug=True, exit_on_err=False)
    
    # Generators and Coroutines
    concolic(test_generators, debug=True, exit_on_err=False)
    concolic(test_generator_expressions, debug=True, exit_on_err=False)
    
    # Dynamic Code Execution
    concolic(test_dynamic_code, debug=True, exit_on_err=False)
    concolic(test_dynamic_function_creation, debug=True, exit_on_err=False)
    
    # Reflection and Introspection
    concolic(test_reflection, debug=True, exit_on_err=False)
    concolic(test_type_annotations, debug=True, exit_on_err=False)
    
    # Advanced Class Features
    concolic(test_descriptors, debug=True, exit_on_err=False)
    concolic(test_slots, debug=True, exit_on_err=False)
    concolic(test_property_decorators, debug=True, exit_on_err=False)
    
    print("\n=== All Advanced Language Features Tests Completed ===") 