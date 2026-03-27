import pytest
import subprocess
import sys

def pytest_sessionstart(session):
    """Run pyright static type checking automatically before executing any tests."""
    print("\n[Pyright] Running static type checks before tests...")
    try:
        result = subprocess.run(
            ["pyright", "sequence_layers"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            # If pyright fails, halt pytest immediately with the output.
            pytest.exit(f"Static type checking failed! Please fix the type errors before running tests.\n\n{result.stdout}")
        else:
            print("[Pyright] Type checks passed successfully.\n")
    except FileNotFoundError:
        print("[Pyright] WARNING: 'pyright' command not found. Skipping static type checks. Please assure pyright is installed (e.g. `pip install pyright`).\n")
