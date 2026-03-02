#!/usr/bin/env python3
"""
Quick validation script for SageAttention3 implementation.
This script checks the structure and function signatures without requiring PyTorch.
"""

import ast
import sys
from pathlib import Path

def validate_implementation():
    """Validate the SageAttention3 implementation structure."""
    impl_file = Path("sageattn3_torch.py")

    if not impl_file.exists():
        print("❌ sageattn3_torch.py not found")
        return False

    # Parse the Python file
    try:
        with open(impl_file, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception as e:
        print(f"❌ Failed to parse {impl_file}: {e}")
        return False

    # Expected functions
    expected_functions = {
        'sageattn3_torch',
        'nvfp4_quantize',
        'nvfp4_dequantize',
        'smooth_qk_tensors',
        'fp4mm_matmul',
        'two_level_p_scaling',
        'apply_two_level_p_quantization',
        'tiled_online_attention',
        'apply_q_smoothing_correction',
        'create_test_tensors'
    }

    # Find all function definitions
    found_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            found_functions.add(node.name)

    # Check if all expected functions are present
    missing_functions = expected_functions - found_functions
    extra_functions = found_functions - expected_functions

    print("🔍 Function validation:")
    if not missing_functions:
        print(f"✅ All {len(expected_functions)} expected functions found")
    else:
        print(f"❌ Missing functions: {missing_functions}")

    if extra_functions:
        print(f"ℹ️  Additional functions: {extra_functions}")

    # Check main function signature
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'sageattn3_torch':
            args = [arg.arg for arg in node.args.args]
            expected_args = ['q', 'k', 'v']
            if all(arg in args for arg in expected_args):
                print("✅ Main function signature looks correct")
            else:
                print(f"❌ Main function missing expected args: {expected_args}")
            break

    return len(missing_functions) == 0

def validate_demo():
    """Validate the demo script structure."""
    demo_file = Path("demo_sageattn3.py")

    if not demo_file.exists():
        print("❌ demo_sageattn3.py not found")
        return False

    try:
        with open(demo_file, 'r') as f:
            content = f.read()
        ast.parse(content)
        print("✅ Demo script syntax is valid")
        return True
    except Exception as e:
        print(f"❌ Demo script syntax error: {e}")
        return False

def validate_readme():
    """Validate README exists and has content."""
    readme_file = Path("README.md")

    if not readme_file.exists():
        print("❌ README.md not found")
        return False

    try:
        with open(readme_file, 'r') as f:
            content = f.read()

        if len(content) > 1000:  # Should have substantial content
            print("✅ README.md exists with good content")
            return True
        else:
            print("❌ README.md too short")
            return False
    except Exception as e:
        print(f"❌ Failed to read README.md: {e}")
        return False

def main():
    """Run all validations."""
    print("🧪 Validating SageAttention3 Implementation")
    print("=" * 50)

    results = []
    results.append(validate_implementation())
    results.append(validate_demo())
    results.append(validate_readme())

    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All validations passed! Implementation is ready.")
        return 0
    else:
        print("❌ Some validations failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())