from typing import List, TypeVar, Any
import random
import string
import math

T = TypeVar('T')

class Randomness:
    """A class to handle various random number and string generation operations."""
    
    def __init__(self, seed: int) -> None:
        """Initialize the random number generator with a seed.
        
        Args:
            seed: Integer value to initialize the random number generator
        """
        self.seed = seed
        random.seed(self.seed)

    def get_random_integer(self, low: int, high: int) -> int:
        """Generate a random integer between low and high (inclusive).
        
        Args:
            low: Lower bound
            high: Upper bound (inclusive)
        
        Returns:
            Random integer between low and high
        """
        if low > high:
            raise ValueError("Low bound must be less than or equal to high bound")
        return random.randint(low, high)

    def get_random_string(self, length: int, use_digits: bool = False) -> str:
        """Generate a random string of specified length.
        
        Args:
            length: Length of the string to generate
            use_digits: Include digits in the generated string
        
        Returns:
            Random string of specified length
        """
        if length < 0:
            raise ValueError("Length must be non-negative")
        chars = string.ascii_letters + (string.digits if use_digits else "")
        return "".join(random.choice(chars) for _ in range(length))

    def get_random_float(self, low: float, high: float) -> float:
        """Generate a random float between low and high.
        
        Args:
            low: Lower bound
            high: Upper bound
        
        Returns:
            Random float between low and high
        """
        if low > high:
            raise ValueError("Low bound must be less than or equal to high bound")
        return random.uniform(low, high)

    def random_choice(self, items: List[T]) -> T:
        """Choose a random item from a list.
        
        Args:
            items: List of items to choose from
        
        Returns:
            Random item from the list
        """
        if not items:
            raise ValueError("List cannot be empty")
        return random.choice(items)

    def shuffle_list(self, items: List[Any]) -> None:
        """Shuffle a list in place.
        
        Args:
            items: List to shuffle
        """
        random.shuffle(items)

    def get_non_prime(self, max_value: int) -> int:
        """Generate a random non-prime number up to max_value.
        
        Args:
            max_value: Upper bound for the generated number
        
        Returns:
            Random non-prime number
        """
        if max_value < 4:
            raise ValueError("Max value must be at least 4")
            
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True

        number = self.get_random_integer(4, max_value)
        while is_prime(number):
            number = self.get_random_integer(4, max_value)
        return number

    def get_seed(self) -> int:
        """Get the current seed value.
        
        Returns:
            Current seed value
        """
        return self.seed
