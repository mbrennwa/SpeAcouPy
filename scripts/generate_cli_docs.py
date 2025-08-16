#!/usr/bin/env python3
"""Generate docs/cli/commands.md from the installed SpeAcouPy CLI.

This script runs:
  - speacoupy --help
  - speacoupy simulate --help   (only if 'simulate' exists)

and writes a Markdown page with both outputs.
It is safe to run locally or in CI. If 'speacoupy' is not on PATH,
the script exits without failing the build (non-zero exit with message).
"""

import subprocess
import shlex
import pathlib
import sys

OUT = pathlib.Path("docs/cli/commands.md")

def run_help(cmd: str) -> str:
    try:
        return subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr
    except FileNotFoundError:
        return ""

def has_subcommand(help_text: str, sub: str) -> bool:
    # naive check: look for the subcommand name in the top-level help text
    return f" {sub} " in help_text or f"\n{sub}\n" in help_text

def main():
    top = run_help("speacoupy --help")
    if not top.strip():
        print("speacoupy not found on PATH. Skipping CLI docs generation.", file=sys.stderr)
        sys.exit(0)  # don't fail the build

    lines = ["# CLI Commands (auto-generated)\n\n"]

    lines += ["## $ speacoupy --help\n\n", "```text\n", top, "\n```\n\n"]

    if has_subcommand(top, "simulate"):
        sim = run_help("speacoupy simulate --help")
        if sim.strip():
            lines += ["## $ speacoupy simulate --help\n\n", "```text\n", sim, "\n```\n\n"]

    OUT.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
