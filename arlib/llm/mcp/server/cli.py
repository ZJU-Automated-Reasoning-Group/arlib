#!/usr/bin/env python3
"""CLI entry point for arlib MCP Z3 server."""

import asyncio
import argparse
import logging
import sys
from .server import serve

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Arlib MCP Z3 Server")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        sys.exit(0)
    except ImportError as e:
        print(f"Error: {e}\nInstall MCP: pip install mcp")
        sys.exit(1)
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
