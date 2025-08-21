"""MCP Server for Z3 solver integration."""

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any

try:
    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.shared.exceptions import McpError
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from ..z3.model_manager import Z3ModelManager
from ..prompts import load_prompt

logger = logging.getLogger(__name__)

def format_model_items(items: list[tuple[int, str]]) -> str:
    """Format model items."""
    return "Model is empty" if not items else "\n".join(f"{i} | {content}" for i, content in items)

def create_server() -> 'Server':
    """Create and configure MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP library not available. Install with: pip install mcp")

    server = Server("arlib-mcp-z3")
    server.capabilities = {"prompts": {}}

    # Initialize Z3 model manager
    model_mgr = Z3ModelManager()
    logger.info("Z3 MCP server initialized")

    # Prompts
    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        """List available prompts."""
        return [
            types.Prompt(name="z3-instructions", description="Z3 usage guide"),
            types.Prompt(name="z3-review", description="Solution verification template"),
        ]

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
        """Get prompt content."""
        try:
            prompt_type = {"z3-instructions": "instructions", "z3-review": "review"}.get(name)
            if not prompt_type:
                raise McpError("INVALID_PARAMS", f"Unknown prompt: {name}")

            content = load_prompt("z3", prompt_type)
            return types.GetPromptResult(
                description=f"Z3 {prompt_type} prompt",
                messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text=content))]
            )
        except Exception as e:
            raise McpError("INTERNAL_ERROR", f"Failed to load prompt: {e}")

    # Tools
    tools = [
        ("clear_model", "Clear the current Z3 model", {}),
        ("add_item", "Add code item at index", {"index": "integer", "content": "string"}),
        ("replace_item", "Replace code item at index", {"index": "integer", "content": "string"}),
        ("delete_item", "Delete code item at index", {"index": "integer"}),
        ("get_model", "Get current model contents", {}),
        ("solve_model", "Solve the Z3 model", {"timeout": "integer"}),
        ("get_solution", "Get current solution", {}),
        ("get_variable_value", "Get variable value", {"variable_name": "string"}),
    ]

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """List available tools."""
        return [
            types.Tool(
                name=name,
                description=desc,
                inputSchema={
                    "type": "object",
                    "properties": {k: {"type": v} for k, v in props.items()},
                    "required": list(props.keys())
                }
            )
            for name, desc, props in tools
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any] | None = None) -> list[types.TextContent]:
        """Handle tool calls."""
        args = arguments or {}

        try:
            # Tool dispatch
            if name == "clear_model":
                result = await model_mgr.clear_model()
            elif name in ["add_item", "replace_item"]:
                method = getattr(model_mgr, name)
                result = await method(args["index"], args["content"])
            elif name == "delete_item":
                result = await model_mgr.delete_item(args["index"])
            elif name == "get_model":
                result = {"success": True, "model": format_model_items(model_mgr.get_model())}
            elif name == "solve_model":
                result = await model_mgr.solve_model(timedelta(seconds=args["timeout"]))
            elif name == "get_solution":
                result = model_mgr.get_solution()
            elif name == "get_variable_value":
                result = model_mgr.get_variable_value(args["variable_name"])
            else:
                raise McpError("METHOD_NOT_FOUND", f"Unknown tool: {name}")

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}, indent=2))]

    return server

async def serve() -> None:
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP library not available. Install with: pip install mcp")

    server = create_server()

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="arlib-mcp-z3",
                server_version="1.0.0",
                capabilities=server.capabilities,
            ),
        )

if __name__ == "__main__":
    asyncio.run(serve())
