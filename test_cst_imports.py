#!/usr/bin/env python
"""Quick test to verify CST imports work"""
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'lob_bench', 'cst_model'))

try:
    import cst
    print("✓ CST module imported")
except Exception as e:
    print(f"✗ Failed to import cst: {e}")
    sys.exit(1)

try:
    from cst import CSTParams, Book
    print("✓ CSTParams and Book imported")
except Exception as e:
    print(f"✗ Failed to import CSTParams/Book: {e}")
    sys.exit(1)

try:
    import param_estimation
    print("✓ param_estimation imported")
except Exception as e:
    print(f"✗ Failed to import param_estimation: {e}")
    sys.exit(1)

try:
    import lobster_conversion
    print("✓ lobster_conversion imported")
except Exception as e:
    print(f"✗ Failed to import lobster_conversion: {e}")
    sys.exit(1)

print("\n✅ All CST imports successful!")

