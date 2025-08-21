# MCP Solver Integration

Z3 solver integration through the MCP (Model Context Protocol) interface for arlib.

## Features

- **Z3 Integration**: Secure Z3 solver with code execution
- **Model Management**: Add/delete/replace code items
- **Templates**: Pre-built Z3 patterns and helper functions
- **MCP Server**: Full MCP server for LLM integration
- **Prompts**: Usage guides and verification templates

## Usage

### Python API
```python
import asyncio
from datetime import timedelta
from arlib.mcp import Z3ModelManager

async def example():
    manager = Z3ModelManager()
    await manager.add_item(0, "x, y = Ints('x y')")
    await manager.add_item(1, "solver = Solver(); solver.add(x + y == 10, x > 0)")
    await manager.add_item(2, "export_solution(solver, {'x': x, 'y': y})")
    result = await manager.solve_model(timedelta(seconds=5))
    print(f"Solution: {result['values']}")

asyncio.run(example())
```

### Templates & Prompts
```python
# Use templates
await manager.add_item(0, "solver, vars = constraint_satisfaction_template()")

# Load prompts
from arlib.mcp import load_prompt
guide = load_prompt("z3", "instructions")
```

### MCP Server
```bash
pip install mcp  # optional
python -m arlib.mcp.server.cli
```

### MCP Config
```json
{"mcpServers": {"arlib-z3": {"command": "python", "args": ["-m", "arlib.mcp.server.cli"]}}}
```

## Components

- `core/`: Base classes and validation
- `z3/`: Z3 solver integration + templates
- `prompts/`: Usage guides and verification
- `server/`: MCP server for LLM integration

## Templates

**Function Templates**: `constraint_satisfaction_template()`, `optimization_template()`, `array_template()`, etc.

**Helper Functions**: `array_is_sorted()`, `all_distinct()`, `exactly_k()`, etc.

## MCP Tools

`clear_model`, `add_item`, `replace_item`, `delete_item`, `get_model`, `solve_model`, `get_solution`, `get_variable_value`

## MCP Prompts

`z3-instructions`: Usage guide | `z3-review`: Solution verification
