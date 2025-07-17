# Minimal Combinatorial Interaction Testing (CIT) Demo
from itertools import product

def demo_cit():
    parameters = {
        "Browser": ["Chrome", "Firefox"],
        "OS": ["Windows", "Linux"],
        "Screen": ["Mobile", "Desktop"]
    }
    keys = list(parameters.keys())
    for idx, values in enumerate(product(*parameters.values())):
        print(f"Test Case {idx+1}: {{ {', '.join(f'{k}: {v}' for k, v in zip(keys, values))} }}")

if __name__ == "__main__":
    demo_cit()