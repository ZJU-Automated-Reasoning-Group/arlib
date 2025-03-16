#!/usr/bin/env python3
"""
Test script for the enhanced SMT server with advanced arlib features.
This script demonstrates how to use the new commands added to the SMT server.

Usage:
1. Run this test script:
   python -m arlib.cli.test_smt_server

The script will automatically start the SMT server if it's not already running.
"""

import os
import time
import sys
import subprocess
import signal
import atexit

# Named pipes for communication
INPUT_PIPE = "/tmp/smt_input"
OUTPUT_PIPE = "/tmp/smt_output"

# Global variable to track the server process
server_process = None


def start_server():
    """Start the SMT server if it's not already running."""
    global server_process
    
    # Check if pipes already exist (server might be running)
    if os.path.exists(INPUT_PIPE) and os.path.exists(OUTPUT_PIPE):
        print("SMT server appears to be running already.")
        return
    
    print("Starting SMT server...")
    try:
        # Start the server as a subprocess
        server_process = subprocess.Popen(
            [sys.executable, "-m", "arlib.cli.smt_server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for pipes to be created
        timeout = 5  # seconds
        start_time = time.time()
        while (not os.path.exists(INPUT_PIPE) or not os.path.exists(OUTPUT_PIPE)):
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for SMT server to start")
            time.sleep(0.1)
        
        # Give the server a moment to initialize
        time.sleep(0.5)
        print("SMT server started successfully.")
    except Exception as e:
        print(f"Failed to start SMT server: {e}")
        sys.exit(1)


def cleanup():
    """Clean up resources when the script exits."""
    global server_process
    
    if server_process:
        print("Shutting down SMT server...")
        try:
            # Try to exit gracefully first
            send_command("exit")
            time.sleep(0.5)
            
            # If still running, terminate
            if server_process.poll() is None:
                server_process.terminate()
                server_process.wait(timeout=2)
                
            print("SMT server shut down.")
        except Exception as e:
            print(f"Error shutting down server: {e}")
            # Force kill if necessary
            if server_process.poll() is None:
                server_process.kill()


def send_command(command):
    """Send a command to the SMT server and return the response."""
    try:
        with open(INPUT_PIPE, 'w') as f:
            f.write(command + '\n')
            f.flush()
        
        # Wait a bit for the server to process the command
        time.sleep(0.1)
        
        with open(OUTPUT_PIPE, 'r') as f:
            response = f.read().strip()
        
        return response
    except Exception as e:
        print(f"Error sending command: {e}")
        return None


def test_basic_functionality():
    """Test basic SMT-LIB2 functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Declare variables
    print("Declaring variables...")
    send_command("declare-const x Int")
    send_command("declare-const y Int")
    send_command("declare-const p Bool")
    
    # Add assertions
    print("Adding assertions...")
    send_command("assert (> x 0)")
    send_command("assert (< y 10)")
    send_command("assert (= (+ x y) 5)")
    
    # Check satisfiability
    print("Checking satisfiability...")
    result = send_command("check-sat")
    print(f"Result: {result}")
    
    # Get model
    print("Getting model...")
    model = send_command("get-model")
    print(f"Model: {model}")
    
    # Get specific values
    print("Getting specific values...")
    values = send_command("get-value x y")
    print(f"Values: {values}")


def test_allsmt():
    """Test AllSMT functionality."""
    print("\n=== Testing AllSMT Functionality ===")
    
    # Reset the solver
    send_command("push")
    send_command("pop")
    
    # Declare variables
    print("Declaring variables...")
    send_command("declare-const a Bool")
    send_command("declare-const b Bool")
    send_command("declare-const c Bool")
    
    # Add assertions
    print("Adding assertions...")
    send_command("assert (or a b c)")
    
    # Set model limit
    print("Setting model limit...")
    send_command("set-option :allsmt-model-limit 10")
    
    # Enumerate all models
    print("Enumerating all models...")
    models = send_command("allsmt a b c")
    print(f"All models: {models}")


def test_unsat_core():
    """Test UNSAT core functionality."""
    print("\n=== Testing UNSAT Core Functionality ===")
    
    # Reset the solver
    send_command("push")
    send_command("pop")
    
    # Declare variables
    print("Declaring variables...")
    send_command("declare-const p Bool")
    send_command("declare-const q Bool")
    send_command("declare-const r Bool")
    
    # Add contradictory assertions
    print("Adding contradictory assertions...")
    send_command("assert p")
    send_command("assert q")
    send_command("assert (not (or p q))")
    
    # Check satisfiability (should be unsat)
    print("Checking satisfiability...")
    result = send_command("check-sat")
    print(f"Result: {result}")
    
    # Get UNSAT core
    print("Getting UNSAT core...")
    core = send_command("unsat-core")
    print(f"UNSAT core: {core}")
    
    # Try with different algorithm
    print("Getting UNSAT core with different algorithm...")
    core = send_command("unsat-core :algorithm=musx")
    print(f"UNSAT core (MUSX): {core}")


def test_backbone():
    """Test backbone literals functionality."""
    print("\n=== Testing Backbone Literals Functionality ===")
    
    # Reset the solver
    send_command("push")
    send_command("pop")
    
    # Declare variables
    print("Declaring variables...")
    send_command("declare-const a Bool")
    send_command("declare-const b Bool")
    
    # Add assertions that make 'a' a backbone literal
    print("Adding assertions...")
    send_command("assert a")
    send_command("assert (or a b)")
    
    # Get backbone literals
    print("Getting backbone literals...")
    backbone = send_command("backbone")
    print(f"Backbone literals: {backbone}")


def test_model_counting():
    """Test model counting functionality."""
    print("\n=== Testing Model Counting Functionality ===")
    
    # Reset the solver
    send_command("push")
    send_command("pop")
    
    # Declare variables
    print("Declaring variables...")
    send_command("declare-const x Bool")
    send_command("declare-const y Bool")
    
    # Add assertions
    print("Adding assertions...")
    send_command("assert (or x y)")
    
    # Count models
    print("Counting models...")
    count = send_command("count-models")
    print(f"Model count: {count}")
    
    # Try approximate counting
    print("Approximate counting...")
    count = send_command("count-models :approximate")
    print(f"Approximate model count: {count}")


def test_help():
    """Test help command."""
    print("\n=== Testing Help Command ===")
    
    help_text = send_command("help")
    print(f"Help text:\n{help_text}")


def main():
    """Run all tests."""
    # Register cleanup function to run on exit
    atexit.register(cleanup)
    
    # Start the server if needed
    start_server()
    
    # Run tests
    test_basic_functionality()
    test_allsmt()
    test_unsat_core()
    test_backbone()
    test_model_counting()
    test_help()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main() 