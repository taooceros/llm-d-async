#!/usr/bin/env python3
"""
Launcher script for RCM Cloud Policy Benchmark Submitter.
Delegates directly to admission_control/experiments/submit_policy_rcm.py.
"""
import os
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TARGET_SCRIPT = SCRIPT_DIR.parent / "admission_control" / "experiments" / "submit_policy_rcm.py"

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, str(TARGET_SCRIPT)] + sys.argv[1:])
