"""
SparseMap module implementation.
This module provides a map implementation that is optimized for sparse mappings,
where most keys map to a "zero" value.
"""

from typing import TypeVar, Generic, Callable, Dict, Iterable, Tuple, List, Iterator, Optional, Protocol, Any


K = TypeVar('K')  # Key type
V = TypeVar('V')  # Value type
R = TypeVar('R')  # Result type


class Zero(Protocol):
    """Protocol for a type that has a zero value and can be compared for equality."""
    
    @staticmethod
    def zero() -> Any:
        """Return the zero value."""
        ...
    
    @staticmethod
    def equal(a: Any, b: Any) -> bool:
        """Check if two values are equal."""
        ...


class SparseMap(Generic[K, V]):
    """
    A map implementation that is optimized for sparse mappings.
    
    In a sparse map, most keys map to a "zero" value, which is not stored
    explicitly. This implementation uses a regular dictionary and only
    stores non-zero values.
    """
    
    def __init__(self, data: Dict[K, V], zero_module):
        """
        Initialize a sparse map.
        
        Args:
            data: The underlying dictionary
            zero_module: Module with zero() and equal() functions
        """
        self._data = data
        self._zero = zero_module
    
    @classmethod
    def empty(cls, zero_module) -> 'SparseMap[K, V]':
        """
        Create an empty sparse map.
        
        Args:
            zero_module: Module with zero() and equal() functions
            
        Returns:
            An empty sparse map
        """
        return cls({}, zero_module)
    
    @classmethod
    def singleton(cls, key: K, value: V, zero_module) -> 'SparseMap[K, V]':
        """
        Create a sparse map with a single key-value pair.
        
        Args:
            key: The key
            value: The value
            zero_module: Module with zero() and equal() functions
            
        Returns:
            A sparse map with a single key-value pair
        """
        if zero_module.equal(value, zero_module.zero()):
            return cls.empty(zero_module)
        return cls({key: value}, zero_module)
    
    def equal(self, other: 'SparseMap[K, V]') -> bool:
        """
        Check if two sparse maps are equal.
        
        Args:
            other: The other sparse map
            
        Returns:
            True if the maps are equal, False otherwise
        """
        if len(self._data) != len(other._data):
            return False
        
        for key, value in self._data.items():
            if key not in other._data or not self._zero.equal(value, other._data[key]):
                return False
        
        return True
    
    def get(self, key: K) -> V:
        """
        Get the value associated with a key.
        
        Args:
            key: The key
            
        Returns:
            The value associated with the key, or the zero value if the key is not in the map
        """
        return self._data.get(key, self._zero.zero())
    
    def set(self, key: K, value: V) -> 'SparseMap[K, V]':
        """
        Set the value associated with a key.
        
        Args:
            key: The key
            value: The value
            
        Returns:
            A new sparse map with the updated key-value pair
        """
        new_data = self._data.copy()
        
        if self._zero.equal(value, self._zero.zero()):
            if key in new_data:
                del new_data[key]
        else:
            new_data[key] = value
        
        return SparseMap(new_data, self._zero)
    
    def is_zero(self) -> bool:
        """
        Check if the map is empty (all values are zero).
        
        Returns:
            True if the map is empty, False otherwise
        """
        return len(self._data) == 0
    
    def merge(self, f: Callable[[K, V, V], V], other: 'SparseMap[K, V]') -> 'SparseMap[K, V]':
        """
        Merge two sparse maps.
        
        Args:
            f: A function that takes a key and two values and returns a new value
            other: The other sparse map
            
        Returns:
            A new sparse map with the merged values
        """
        result = {}
        
        # Collect all keys
        all_keys = set(self._data.keys()) | set(other._data.keys())
        
        for key in all_keys:
            v1 = self.get(key)
            v2 = other.get(key)
            merged = f(key, v1, v2)
            
            if not self._zero.equal(merged, self._zero.zero()):
                result[key] = merged
        
        return SparseMap(result, self._zero)
    
    def modify(self, key: K, f: Callable[[V], V]) -> 'SparseMap[K, V]':
        """
        Modify the value associated with a key.
        
        Args:
            key: The key
            f: A function that takes the old value and returns a new value
            
        Returns:
            A new sparse map with the modified value
        """
        old_value = self.get(key)
        new_value = f(old_value)
        return self.set(key, new_value)
    
    def map(self, f: Callable[[K, V], V]) -> 'SparseMap[K, V]':
        """
        Apply a function to each key-value pair.
        
        Args:
            f: A function that takes a key and a value and returns a new value
            
        Returns:
            A new sparse map with the mapped values
        """
        result = {}
        
        for key, value in self._data.items():
            new_value = f(key, value)
            if not self._zero.equal(new_value, self._zero.zero()):
                result[key] = new_value
        
        return SparseMap(result, self._zero)
    
    def extract(self, key: K) -> Tuple[V, 'SparseMap[K, V]']:
        """
        Extract a key-value pair from the map.
        
        Args:
            key: The key
            
        Returns:
            A tuple of the value and the map without the key
        """
        value = self.get(key)
        
        new_data = self._data.copy()
        if key in new_data:
            del new_data[key]
        
        return value, SparseMap(new_data, self._zero)
    
    def items(self) -> Iterator[Tuple[K, V]]:
        """
        Iterate over the key-value pairs.
        
        Returns:
            An iterator over the key-value pairs
        """
        return iter(self._data.items())
    
    @classmethod
    def from_items(cls, items: Iterable[Tuple[K, V]], zero_module) -> 'SparseMap[K, V]':
        """
        Create a sparse map from an iterable of key-value pairs.
        
        Args:
            items: The key-value pairs
            zero_module: Module with zero() and equal() functions
            
        Returns:
            A sparse map from the key-value pairs
        """
        result = cls.empty(zero_module)
        
        for key, value in items:
            result = result.set(key, value)
        
        return result
    
    def fold(self, f: Callable[[K, V, R], R], init: R) -> R:
        """
        Fold over the key-value pairs.
        
        Args:
            f: A function that takes a key, a value, and an accumulator and returns a new accumulator
            init: The initial accumulator
            
        Returns:
            The final accumulator
        """
        result = init
        
        for key, value in self._data.items():
            result = f(key, value, result)
        
        return result
    
    def min_support(self) -> Tuple[K, V]:
        """
        Get the key-value pair with the minimum key.
        
        Returns:
            The key-value pair with the minimum key
            
        Raises:
            ValueError: If the map is empty
        """
        if not self._data:
            raise ValueError("min_support called on empty map")
        
        min_key = min(self._data.keys())
        return min_key, self._data[min_key]
    
    def pop(self) -> Tuple[Tuple[K, V], 'SparseMap[K, V]']:
        """
        Pop the key-value pair with the minimum key.
        
        Returns:
            A tuple of the key-value pair and the map without the key
            
        Raises:
            ValueError: If the map is empty
        """
        if not self._data:
            raise ValueError("pop called on empty map")
        
        min_key = min(self._data.keys())
        value = self._data[min_key]
        
        new_data = self._data.copy()
        del new_data[min_key]
        
        return (min_key, value), SparseMap(new_data, self._zero)
    
    def hash(self, f: Callable[[Tuple[K, V]], int]) -> int:
        """
        Hash the sparse map.
        
        Args:
            f: A function that takes a key-value pair and returns a hash
            
        Returns:
            A hash of the sparse map
        """
        return hash(tuple(f((k, v)) for k, v in sorted(self._data.items())))
    
    def compare(self, f: Callable[[V, V], int], other: 'SparseMap[K, V]') -> int:
        """
        Compare two sparse maps.
        
        Args:
            f: A function that compares two values
            other: The other sparse map
            
        Returns:
            A negative number if self < other, 0 if self == other, a positive number if self > other
        """
        # Compare lengths first
        if len(self._data) < len(other._data):
            return -1
        elif len(self._data) > len(other._data):
            return 1
        
        # Compare keys
        self_keys = sorted(self._data.keys())
        other_keys = sorted(other._data.keys())
        
        for self_key, other_key in zip(self_keys, other_keys):
            if self_key < other_key:
                return -1
            elif self_key > other_key:
                return 1
        
        # Compare values
        for key in self_keys:
            cmp = f(self._data[key], other._data[key])
            if cmp != 0:
                return cmp
        
        return 0


# Example of how to use SparseMap
if __name__ == "__main__":
    # Define a zero module for integers
    class IntZero:
        @staticmethod
        def zero():
            return 0
        
        @staticmethod
        def equal(a, b):
            return a == b
    
    # Create some sparse maps
    empty = SparseMap.empty(IntZero)
    map1 = SparseMap.singleton("a", 1, IntZero)
    map2 = SparseMap.singleton("b", 2, IntZero)
    
    # Combine maps
    map3 = map1.set("c", 3)
    
    # Merge maps with addition
    add = lambda k, v1, v2: v1 + v2
    map4 = map1.merge(add, map2)
    
    # Print out maps
    print("map1:", list(map1.items()))  # [('a', 1)]
    print("map2:", list(map2.items()))  # [('b', 2)]
    print("map3:", list(map3.items()))  # [('a', 1), ('c', 3)]
    print("map4:", list(map4.items()))  # [('a', 1), ('b', 2)]
    
    # Extract a value
    value, map5 = map3.extract("a")
    print("Extracted:", value)  # 1
    print("map5:", list(map5.items()))  # [('c', 3)]
    
    # Fold to compute sum of values
    total = map3.fold(lambda k, v, acc: acc + v, 0)
    print("Sum of values in map3:", total)  # 4 