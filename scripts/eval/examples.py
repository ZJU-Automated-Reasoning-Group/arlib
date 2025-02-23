# coding: utf-8
import numpy as np
from typing import List, Tuple

from plotting import ScatterPlot, CactusPlot, BoxPlot

def generate_scatter_data() -> Tuple[List[float], List[float]]:
    """Generate sample data for scatter plot."""
    return ([0.4, 0.6, 0.33, 0.2], [0.3, 0.7, 0.6, 0.33])

def generate_multi_group_data() -> Tuple[Tuple[List[float], List[float]], 
                                       Tuple[List[float], List[float]]]:
    """Generate sample data for multi-group scatter plot."""
    sat_data = ([2, 3, 4, 8], [1, 5, 6, 6])
    unsat_data = ([2, 3, 4, 5], [1, 2, 6, 8])
    return (sat_data, unsat_data)

def generate_cactus_data() -> List[List[float]]:
    """Generate sample data for cactus plot."""
    return [
        [1, 2, 3, 3.5, 2.5, 4, 6],
        [5, 2, 2, 5, 8, 2, 10],
        [1, 7, 3, 3, 2, 4, 10, 12, 15],
        [5, 6, 1, 2, 3, 8, 7, 9, 12, 10]
    ]

def generate_box_data() -> List[List[float]]:
    """Generate sample data for box plot."""
    return [
        [1, 2, 3, 4, 3, 5, 2],
        [1, 2, 3, 4, 3, 5, 2],
        [1, 2, 3, 4, 3, 5, 2]
    ]

def generate_multi_group_box_data() -> List[List[List[float]]]:
    """Generate sample data for multi-group box plot."""
    return [
        [[1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]],
        [[2, 2, 3, 4, 3, 5, 6], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]],
        [[2, 2, 3, 4, 3, 5, 6], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]]
    ]

def test_scatter_plot():
    """Demonstrate basic scatter plot usage."""
    data = generate_scatter_data()
    plotter = ScatterPlot("Tool A", "Tool B")
    plotter.plot(data)

def test_scatter_plot_multi_groups():
    """Demonstrate scatter plot with multiple groups."""
    data = generate_multi_group_data()
    plotter = ScatterPlot("SAT Solver 1", "SAT Solver 2")
    plotter.plot(data, multi_group=True)

def test_cactus_plot():
    """Demonstrate cactus plot usage."""
    data = generate_cactus_data()
    plotter = CactusPlot()
    plotter.plot(data)

def test_box_plot():
    """Demonstrate box plot usage."""
    data = generate_box_data()
    plotter = BoxPlot()
    plotter.plot(data)

def test_box_plot_multi_groups():
    """Demonstrate box plot with multiple groups."""
    data = generate_multi_group_box_data()
    plotter = BoxPlot()
    plotter.plot(data, multi_group=True)

if __name__ == "__main__":
    # Uncomment tests to run
    test_scatter_plot()
    # test_scatter_plot_multi_groups()
    # test_cactus_plot()
    # test_box_plot()
    # test_box_plot_multi_groups()