"""
Test runner script for heart disease prediction tests
"""
import sys
import os
import subprocess
import argparse

def run_tests(test_type="all", verbose=False, coverage=False):
    """
    Run tests with specified options
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'all')
        verbose: Enable verbose output
        coverage: Generate coverage report
    """
    # Change to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", "tests/"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    # Add test type filter
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code: {e.returncode}")
        return e.returncode


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run heart disease prediction tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "all", "slow"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    return run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage
    )


if __name__ == "__main__":
    sys.exit(main())
