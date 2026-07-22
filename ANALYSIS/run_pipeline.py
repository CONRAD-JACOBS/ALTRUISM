#!/usr/bin/python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

"""
python3 run_pipeline.py
That runs scripts 1-6 and uses 3_purified.csv downstream.
"""

HERE = Path(__file__).resolve().parent
PURIFIED_CSV = HERE / "3_purified.csv"
DFBETAS_SENSITIVITY_CSV = HERE / "3_dfbetas_sensitivity.csv"

# When True, archive files from numbered output directories (for example,
# 3_purify and 5_nb_models) in the operating-system Trash before a new run.
# The directories themselves are retained and recreated by their stage scripts.
CLEAR_NUMBERED_OUTPUT_FOLDERS = True


def _trash_path(path):
    """Move a path to the OS Trash without silently overwriting an older item."""
    try:
        from send2trash import send2trash
        send2trash(str(path))
        return
    except ImportError:
        pass

    # Dependency-free fallback for macOS, where this project is run.
    trash = Path.home() / ".Trash"
    trash.mkdir(parents=True, exist_ok=True)
    destination = trash / path.name
    if destination.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        destination = trash / "{}_{}{}".format(path.stem, stamp, path.suffix)
        counter = 2
        while destination.exists():
            destination = trash / "{}_{}_{}{}".format(path.stem, stamp, counter, path.suffix)
            counter += 1
    shutil.move(str(path), str(destination))


def clear_numbered_output_folders():
    numbered_folders = sorted(
        path for path in HERE.iterdir()
        if path.is_dir() and re.match(r"^\d+_", path.name)
    )
    moved = 0
    for folder in numbered_folders:
        for item in folder.iterdir():
            _trash_path(item)
            moved += 1
    print(
        "Cleanup enabled: moved {} item(s) from {} numbered output folder(s) to Trash.".format(
            moved, len(numbered_folders)
        ),
        flush=True,
    )


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

    if CLEAR_NUMBERED_OUTPUT_FOLDERS:
        clear_numbered_output_folders()
    else:
        print("Cleanup disabled: existing numbered-folder outputs retained.", flush=True)

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
