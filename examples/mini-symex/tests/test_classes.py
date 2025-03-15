#!/usr/bin/env python3
"""
Test file for classes and objects in the mini-symex framework.
"""

import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *
import math

class Shape:
    """Base class for shapes"""
    def __init__(self, name):
        self.name = name
    
    def area(self):
        return 0
    
    def describe(self):
        return f"This is a {self.name} with area {self.area()}"

class Rectangle(Shape):
    """Rectangle class that inherits from Shape"""
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    """Circle class that inherits from Shape"""
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius * self.radius

def test_classes_and_objects():
    """Test classes and objects"""
    print("\n=== Testing Classes and Objects ===")
    
    # Create symbolic values for dimensions
    width = mk_int("width")
    height = mk_int("height")
    radius = mk_int("radius")
    
    # Create objects with symbolic values
    rect = mk_object("rect", Rectangle, width, height)
    circle = mk_object("circle", Circle, radius)
    
    # Test method calls and polymorphism
    rect_area = rect.area()
    circle_area = circle.area()
    
    # Convert circle_area to int to avoid Z3 sort mismatch
    circle_area_int = int(circle_area)
    
    if rect_area > circle_area_int:
        print(f"Rectangle area > Circle area: width={width}, height={height}, radius={radius}")
    else:
        print(f"Rectangle area <= Circle area: width={width}, height={height}, radius={radius}")
    
    # Test inheritance
    rect_desc = rect.describe()
    circle_desc = circle.describe()
    
    print(f"Rectangle description: {rect_desc}")
    print(f"Circle description: {circle_desc}")

if __name__ == "__main__":
    print("=== Running Classes and Objects Test ===")
    concolic(test_classes_and_objects, debug=True, exit_on_err=False)
    print("\n=== Test Completed ===") 