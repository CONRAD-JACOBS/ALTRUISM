#!/usr/bin/python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

"""
python3 run_pipeline.py
That skips script 3 and uses 2_simplified.csv downstream.
python3 run_pipeline.py --with-exclusions
That runs script 3 and makes scripts 4/5 use 3_purified.csv.
"""

HERE = Path(__file__).resolve().parent
SIMPLIFIED_CSV = HERE / "2_simplified.csv"
PURIFIED_CSV = HERE / "3_purified.csv"


def run_script(script_name, env=None):
    script_path = HERE / script_name
    print("\n=== Running {} ===".format(script_name), flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=str(HERE), env=env, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the ALTRUISM analysis scripts in sequence."
    )
    parser.add_argument(
        "--with-exclusions",
        action="store_true",
        help="Run 3_exclude.py and use 3_purified.csv for downstream scripts.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    downstream_input = PURIFIED_CSV if args.with_exclusions else SIMPLIFIED_CSV
    env["ANALYSIS_INPUT_CSV"] = str(downstream_input)

    run_script("1_assemble.py")
    run_script("2_simplify.py")

    if args.with_exclusions:
        run_script("3_exclude.py")
    else:
        print("\n=== Skipping 3_exclude.py; downstream scripts will use {} ===".format(SIMPLIFIED_CSV))

    print("Downstream input: {}".format(downstream_input))
    run_script("4_descriptives.py", env=env)
    run_script("5_nb_model.py", env=env)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
