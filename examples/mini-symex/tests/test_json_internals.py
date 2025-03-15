#!/usr/bin/env python3
"""
Advanced test file for the Python JSON library internals using the mini-symex framework.
This test directly targets the internal implementation details of the JSON library.
"""

import sys
import os
import json
import inspect
import time
import re

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concolic import *

# Import internal JSON modules
try:
    from json import decoder, encoder, scanner
except ImportError:
    print("Could not import internal JSON modules")

# Test the internal string parsing functions in the decoder
def test_decoder_internals():
    """Test the internal string parsing functions in the decoder"""
    print("\n=== Testing Decoder Internals ===")
    
    # Create a symbolic string
    s = mk_str("s")
    
    # Test the internal string parsing function
    if hasattr(decoder, 'scanstring'):
        try:
            # scanstring(string, end, strict=True)
            # Returns a tuple of (parsed_string, index)
            if s.startswith('"') and '"' in s[1:]:
                # Find the closing quote
                end_quote = s.find('"', 1)
                if end_quote > 0:
                    try:
                        result, index = decoder.scanstring(s, 1)
                        print(f"scanstring result: {result}, index: {index}")
                        print(f"Input string: {s}")
                    except Exception as e:
                        print(f"scanstring error: {e}")
        except Exception as e:
            print(f"Error testing scanstring: {e}")
    
    # Test the internal number parsing
    if hasattr(decoder, 'JSONDecoder'):
        try:
            # Create a decoder
            dec = decoder.JSONDecoder()
            
            # Test parse_int and parse_float if available
            if hasattr(dec, 'parse_int') and callable(dec.parse_int):
                if s.isdigit():
                    try:
                        result = dec.parse_int(s)
                        print(f"parse_int result: {result}")
                    except Exception as e:
                        print(f"parse_int error: {e}")
            
            if hasattr(dec, 'parse_float') and callable(dec.parse_float):
                if '.' in s and s.replace('.', '').isdigit():
                    try:
                        result = dec.parse_float(s)
                        print(f"parse_float result: {result}")
                    except Exception as e:
                        print(f"parse_float error: {e}")
        except Exception as e:
            print(f"Error testing decoder methods: {e}")

# Test the internal scanner functions
def test_scanner_internals():
    """Test the internal scanner functions"""
    print("\n=== Testing Scanner Internals ===")
    
    # Create a symbolic string
    s = mk_str("s")
    
    # Test the py_make_scanner function if available
    if hasattr(scanner, 'py_make_scanner'):
        try:
            # Create a decoder
            dec = decoder.JSONDecoder()
            
            # Create a scanner
            scan_once = scanner.py_make_scanner(dec)
            
            # Test the scanner with different string patterns
            if s.startswith('{'):
                try:
                    value, end = scan_once(s, 0)
                    print(f"Scanner result for object: {value}, end: {end}")
                except Exception as e:
                    print(f"Scanner error for object: {e}")
            
            elif s.startswith('['):
                try:
                    value, end = scan_once(s, 0)
                    print(f"Scanner result for array: {value}, end: {end}")
                except Exception as e:
                    print(f"Scanner error for array: {e}")
            
            elif s.startswith('"'):
                try:
                    value, end = scan_once(s, 0)
                    print(f"Scanner result for string: {value}, end: {end}")
                except Exception as e:
                    print(f"Scanner error for string: {e}")
            
            elif s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                try:
                    value, end = scan_once(s, 0)
                    print(f"Scanner result for number: {value}, end: {end}")
                except Exception as e:
                    print(f"Scanner error for number: {e}")
        except Exception as e:
            print(f"Error testing scanner: {e}")

# Test the internal encoder functions
def test_encoder_internals():
    """Test the internal encoder functions"""
    print("\n=== Testing Encoder Internals ===")
    
    # Create symbolic values
    x = mk_int("x")
    s = mk_str("s")
    
    # Test the internal encoding functions
    if hasattr(encoder, 'JSONEncoder'):
        try:
            # Create an encoder
            enc = encoder.JSONEncoder()
            
            # Test encoding different types
            try:
                # Test encoding an integer
                result = enc.encode(x)
                print(f"Encoded integer: {x} -> {result}")
                
                # Test encoding a string
                result = enc.encode(s)
                print(f"Encoded string: {s} -> {result}")
                
                # Test encoding a list
                result = enc.encode([x, s])
                print(f"Encoded list: [{x}, {s}] -> {result}")
                
                # Test encoding a dictionary
                result = enc.encode({"num": x, "str": s})
                print(f"Encoded dict: {{num: {x}, str: {s}}} -> {result}")
            except Exception as e:
                print(f"Encoding error: {e}")
            
            # Test the iterencode method
            try:
                # Test iterencode with a complex object
                obj = {"num": x, "list": [1, 2, x], "nested": {"str": s}}
                chunks = list(enc.iterencode(obj))
                result = ''.join(chunks)
                print(f"iterencode chunks: {chunks}")
                print(f"iterencode result: {result}")
            except Exception as e:
                print(f"iterencode error: {e}")
        except Exception as e:
            print(f"Error testing encoder: {e}")

# Test the internal escape handling in the encoder
def test_string_escape_handling():
    """Test the internal string escape handling in the encoder"""
    print("\n=== Testing String Escape Handling ===")
    
    # Create a symbolic string with potential escape characters
    s = mk_str("s")
    
    # Test encoding strings with special characters
    if hasattr(encoder, 'encode_basestring') and callable(encoder.encode_basestring):
        try:
            # Test the encode_basestring function
            result = encoder.encode_basestring(s)
            print(f"encode_basestring: {s} -> {result}")
            
            # Check if the result is properly escaped
            if result.startswith('"') and result.endswith('"'):
                print("Result is properly quoted")
                
                # Check for common escape sequences
                if '\\' in result:
                    print("Result contains escape sequences")
        except Exception as e:
            print(f"encode_basestring error: {e}")
    
    # Test the ASCII version if available
    if hasattr(encoder, 'encode_basestring_ascii') and callable(encoder.encode_basestring_ascii):
        try:
            # Test the encode_basestring_ascii function
            result = encoder.encode_basestring_ascii(s)
            print(f"encode_basestring_ascii: {s} -> {result}")
            
            # Check if the result contains only ASCII characters
            if all(ord(c) < 128 for c in result):
                print("Result contains only ASCII characters")
        except Exception as e:
            print(f"encode_basestring_ascii error: {e}")

# Test the internal whitespace handling in the decoder
def test_whitespace_handling():
    """Test the internal whitespace handling in the decoder"""
    print("\n=== Testing Whitespace Handling ===")
    
    # Create a symbolic string with potential whitespace
    s = mk_str("s")
    
    # Add constraints to make it more likely to have whitespace
    if ' ' in s or '\\t' in s or '\\n' in s or '\\r' in s:
        # Test parsing with whitespace
        try:
            result = json.loads(s)
            print(f"Successfully parsed with whitespace: {s} -> {result}")
        except json.JSONDecodeError as e:
            # Check if the error message mentions whitespace
            if "whitespace" in str(e).lower():
                print(f"Whitespace error: {e}")
            else:
                print(f"Other error: {e}")
    
    # Test with a specific pattern: whitespace + valid JSON
    if s.strip() and s != s.strip():
        # There is whitespace at the beginning or end
        stripped = s.strip()
        try:
            # Try to parse the stripped version
            result = json.loads(stripped)
            print(f"Successfully parsed stripped version: {stripped} -> {result}")
            
            # Now try the original with whitespace
            try:
                result2 = json.loads(s)
                print(f"Successfully parsed with whitespace: {s} -> {result2}")
                
                # Compare results
                if result == result2:
                    print("Results match with and without whitespace")
            except json.JSONDecodeError as e:
                print(f"Error parsing with whitespace: {e}")
        except json.JSONDecodeError:
            # If the stripped version can't be parsed, skip the test
            pass

# Test the internal error handling in the decoder
def test_error_handling():
    """Test the internal error handling in the decoder"""
    print("\n=== Testing Error Handling ===")
    
    # Create a symbolic string
    s = mk_str("s")
    
    # Test with various error patterns
    try:
        result = json.loads(s)
        print(f"Successfully parsed: {s} -> {result}")
    except json.JSONDecodeError as e:
        # Analyze the error message
        error_msg = str(e)
        print(f"JSONDecodeError: {error_msg}")
        
        # Check for specific error patterns
        if "Expecting value" in error_msg:
            print("Error type: Expecting value")
            # Extract the position information
            match = re.search(r"line (\d+) column (\d+)", error_msg)
            if match:
                line, col = match.groups()
                print(f"Error position: line {line}, column {col}")
        
        elif "Expecting property name" in error_msg:
            print("Error type: Expecting property name")
        
        elif "Expecting ':'" in error_msg:
            print("Error type: Expecting colon")
        
        elif "Expecting ',' or" in error_msg:
            print("Error type: Expecting comma or closing bracket/brace")
        
        elif "Extra data" in error_msg:
            print("Error type: Extra data")
        
        elif "Unterminated string" in error_msg:
            print("Error type: Unterminated string")
        
        else:
            print(f"Unknown error type: {error_msg}")

def run_json_internals_tests():
    """Run all JSON internals tests"""
    start_time = time.time()
    
    print("\n=== Running Decoder Internals Test ===")
    crashes, stats = concolic(test_decoder_internals, debug=True, exit_on_err=False, max_states=300, max_paths=300)
    print(f"\nDecoder internals test completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    print("\n=== Running Scanner Internals Test ===")
    crashes, stats = concolic(test_scanner_internals, debug=True, exit_on_err=False, max_states=300, max_paths=300)
    print(f"\nScanner internals test completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    print("\n=== Running Encoder Internals Test ===")
    crashes, stats = concolic(test_encoder_internals, debug=True, exit_on_err=False, max_states=300, max_paths=300)
    print(f"\nEncoder internals test completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    print("\n=== Running String Escape Handling Test ===")
    crashes, stats = concolic(test_string_escape_handling, debug=True, exit_on_err=False, max_states=300, max_paths=300)
    print(f"\nString escape handling test completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    print("\n=== Running Whitespace Handling Test ===")
    crashes, stats = concolic(test_whitespace_handling, debug=True, exit_on_err=False, max_states=25)
    print(f"\nWhitespace handling test completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    print("\n=== Running Error Handling Test ===")
    crashes, stats = concolic(test_error_handling, debug=True, exit_on_err=False, max_states=30)
    print(f"\nError handling test completed with {stats['states_explored']} states and {stats['paths_explored']} paths explored")
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    print("=== Testing Python JSON Library Internals with Concolic Execution ===")
    run_json_internals_tests()
    print("\n=== All JSON Internals Tests Completed ===") 