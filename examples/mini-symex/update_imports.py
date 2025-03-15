#!/usr/bin/env python3
"""
Script to update import statements in test files that have been moved to the tests directory.
This adds parent directory to sys.path to allow importing modules from the parent directory.
"""

import os
import re
import sys

def update_imports(file_path):
    """
    Update import statements in a test file to account for its new location.
    Adds a sys.path.append statement to allow importing from the parent directory.
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file already has sys.path manipulation
    if "sys.path.append" in content or "sys.path.insert" in content:
        # Remove the existing sys.path manipulation and module imports to reposition them
        content = re.sub(r'import\s+sys\s*\n', '', content)
        content = re.sub(r'import\s+os\s*\n', '', content)
        content = re.sub(r'sys\.path\.append\(.*?\)\s*\n', '', content)
        
        # Remove existing module imports to reposition them
        module_imports = []
        import_pattern = r'(from\s+(\w+)\s+import\s+.*?\n|import\s+(\w+).*?\n)'
        
        for match in re.finditer(import_pattern, content):
            if not match.group(0).startswith(('import sys', 'import os')):
                module_imports.append(match.group(0))
                content = content.replace(match.group(0), '', 1)
    else:
        # If no sys.path manipulation, just collect module imports
        module_imports = []
        import_pattern = r'(from\s+(\w+)\s+import\s+.*?\n|import\s+(\w+).*?\n)'
        
        for match in re.finditer(import_pattern, content):
            if not match.group(0).startswith(('import sys', 'import os')):
                module_imports.append(match.group(0))
                content = content.replace(match.group(0), '', 1)
    
    # Prepare the new import block
    import_block = "import sys\nimport os\n\n# Add parent directory to path to allow importing modules\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n"
    
    # Add back the module imports
    for imp in module_imports:
        import_block += imp
    
    # Now insert the import block at the appropriate position
    shebang_pattern = r'(#!/usr/bin/env python3\s*\n)'
    docstring_pattern = r'(""".*?"""\s*\n)'
    
    if re.search(shebang_pattern, content):
        # After shebang
        if re.search(docstring_pattern, content, re.DOTALL):
            # After docstring if it exists
            content = re.sub(docstring_pattern, r'\1\n' + import_block, content, count=1, flags=re.DOTALL)
        else:
            # After shebang if no docstring
            content = re.sub(shebang_pattern, r'\1\n' + import_block, content, count=1)
    else:
        # At the beginning if no shebang
        if re.search(docstring_pattern, content, re.DOTALL):
            # After docstring if it exists
            content = re.sub(docstring_pattern, r'\1\n' + import_block, content, count=1, flags=re.DOTALL)
        else:
            # At the very beginning
            content = import_block + content
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated imports in {file_path}")

def main():
    """Update imports in all test files in the tests directory."""
    tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    
    for filename in os.listdir(tests_dir):
        if filename.endswith('.py'):
            file_path = os.path.join(tests_dir, filename)
            update_imports(file_path)

if __name__ == "__main__":
    main() 