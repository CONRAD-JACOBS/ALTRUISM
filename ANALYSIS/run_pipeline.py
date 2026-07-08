#!/usr/bin/python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

"""
python3 run_pipeline.py
That runs scripts 1-6 and uses 3_purified.csv downstream.
"""

HERE = Path(__file__).resolve().parent
PURIFIED_CSV = HERE / "3_purified.csv"
DFBETAS_SENSITIVITY_CSV = HERE / "3_dfbetas_sensitivity.csv"


def run_script(script_name, env=None):
    script_path = HERE / script_name
    print("\n=== Running {} ===".format(script_name), flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=str(HERE), env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the ALTRUISM analysis scripts in sequence."
    )
    parser.parse_args()

    env = os.environ.copy()
    env["ANALYSIS_INPUT_CSV"] = str(PURIFIED_CSV)
    env["ANALYSIS_SENSITIVITY_INPUT_CSV"] = str(DFBETAS_SENSITIVITY_CSV)

    run_script("1_assemble.py")
    run_script("2_simplify.py")
    run_script("3_purify.py")

    print("Downstream input: {}".format(PURIFIED_CSV))
    run_script("4_descriptives.py", env=env)
    run_script("5_nb_model.py", env=env)
    run_script("6_correlations.py", env=env)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
