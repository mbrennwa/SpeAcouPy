import subprocess, sys, pathlib

def test_import():
    import speacoupy as pkg
    assert hasattr(pkg, "ResponseSolver")

def test_cli_help():
    # Just check the CLI runs and prints usage
    cmd = [sys.executable, "-m", "speacoupy.cli", "--help"]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    assert cp.returncode == 0
    assert "YAML" in cp.stdout or "config" in cp.stdout
