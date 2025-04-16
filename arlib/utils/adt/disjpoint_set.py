"""
Disjoint Set data structure implementation in Python.
Equivalent to the OCaml DisjointSet module in SRK.
"""

from typing import Dict, Optional, TypeVar, Generic, Callable, Set, Any, Hashable


T = TypeVar('T', bound=Hashable)
S = TypeVar('S')


class DisjointSet(Generic[T]):
    """
    A disjoint-set (union-find) data structure that efficiently keeps track of a partition of elements into disjoint subsets.
    """
    
    class _Set:
        """Internal representation of a set in the disjoint-set structure."""
        def __init__(self, id_: int):
            self.id = id_
            self.parent = None  # type: Optional[DisjointSet._Set]
            self.rank = 0
    
    def __init__(self, initial_size: int = 16):
        """
        Initialize an empty disjoint-set data structure.
        
        Args:
            initial_size: Initial size of the hash table
        """
        self.set_map: Dict[T, DisjointSet._Set] = {}
        self.size = 0
    
    def copy(self) -> 'DisjointSet[T]':
        """
        Create a copy of this disjoint-set structure.
        
        Returns:
            A new DisjointSet containing the same elements and partitions
        """
        new_ds = DisjointSet()
        new_ds.set_map = self.set_map.copy()
        new_ds.size = self.size
        return new_ds
    
    def _find_impl(self, set_: _Set) -> _Set:
        """
        Find the representative of a set with path compression.
        
        Args:
            set_: The set to find the representative for
            
        Returns:
            The representative (root) of the set
        """
        if set_.parent is None:
            return set_
        else:
            root = self._find_impl(set_.parent)
            set_.parent = root  # Path compression
            return root
    
    def find(self, elem: T) -> _Set:
        """
        Find the representative set for an element, creating a new set if needed.
        
        Args:
            elem: The element to find
            
        Returns:
            The representative set for the element
        """
        try:
            set_ = self.set_map[elem]
        except KeyError:
            # Add elem to the disjoint set structure
            set_ = self._Set(self.size)
            self.set_map[elem] = set_
            self.size += 1
        
        return self._find_impl(set_)
    
    @staticmethod
    def eq(x: _Set, y: _Set) -> bool:
        """
        Check if two sets are equal (have the same representative).
        
        Args:
            x: First set
            y: Second set
            
        Returns:
            True if the sets are equal, False otherwise
        """
        return x.id == y.id
    
    def union(self, x: _Set, y: _Set) -> _Set:
        """
        Union two sets by rank.
        
        Args:
            x: First set
            y: Second set
            
        Returns:
            The representative of the merged set
        """
        x_root = self._find_impl(x)
        y_root = self._find_impl(y)
        
        if x_root.rank > y_root.rank:
            y_root.parent = x_root
        elif x_root.rank < y_root.rank:
            x_root.parent = y_root
        elif not self.eq(x_root, y_root):
            y_root.parent = x_root
            x_root.rank += 1
        
        return self._find_impl(x)
    
    def same_set(self, x: T, y: T) -> bool:
        """
        Check if two elements belong to the same set.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if the elements are in the same set, False otherwise
        """
        return self.eq(self.find(x), self.find(y))
    
    def reverse_map(self, empty: S, add: Callable[[T, S], S]) -> Callable[[T], S]:
        """
        Create a mapping from elements to their representative sets.
        
        Args:
            empty: An empty value for the mapping
            add: A function to add an element to a mapping
            
        Returns:
            A function that maps elements to their representative set
        """
        # Create a dictionary mapping representatives to their elements
        rep_map = {}
        
        for m, set_ in self.set_map.items():
            rep = self._find_impl(set_)
            rep_id = rep.id
            
            if rep_id not in rep_map:
                rep_map[rep_id] = empty
            
            rep_map[rep_id] = add(m, rep_map[rep_id])
        
        # Return a function that looks up an element's representative
        def lookup(m: T) -> S:
            set_ = self.set_map[m]
            rep = self._find_impl(set_)
            return rep_map[rep.id]
        
        return lookup
    
    def clear(self) -> None:
        """Clear the disjoint-set structure, removing all elements."""
        self.set_map.clear()
        self.size = 0


# Usage example:
if __name__ == "__main__":
    # Create a disjoint-set structure
    ds = DisjointSet[str]()
    
    # Find or create sets for elements
    a_set = ds.find("a")
    b_set = ds.find("b")
    c_set = ds.find("c")
    d_set = ds.find("d")
    
    # Union some sets
    ds.union(a_set, b_set)  # a and b are now in the same set
    ds.union(c_set, d_set)  # c and d are now in the same set
    
    # Check if elements are in the same set
    print(f"a and b in same set: {ds.same_set('a', 'b')}")  # True
    print(f"a and c in same set: {ds.same_set('a', 'c')}")  # False
    
    # Union more sets
    ds.union(a_set, c_set)  # Now all elements are in the same set
    
    print(f"a and d in same set: {ds.same_set('a', 'd')}")  # True
    
    # Create a reverse mapping
    empty_set = set()
    add_func = lambda elem, s: s.union({elem})
    mapping = ds.reverse_map(empty_set, add_func)
    
    # Look up the set containing 'a'
    print(f"Elements in a's set: {mapping('a')}")  # Should contain a, b, c, d 