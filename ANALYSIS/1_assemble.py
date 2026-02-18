#!/usr/bin/python3
import pandas as pd
import os, glob
from pathlib import Path

root = Path(__file__).resolve().parents[1]

def merge_csvs(input_folder, output_file):
    if not os.path.isdir(input_folder):
        raise ValueError(f"Folder does not exist: {input_folder}")

    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    if not csv_files:
        raise ValueError(f"No CSV files found in: {input_folder}")

    df_list = []
    for file in csv_files:
        df = pd.read_csv(file, dtype=str)
        print("IN:", os.path.basename(file), "shape:", df.shape, "first cols:", df.columns[:3].tolist())
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully merged {len(csv_files)} CSV files into {output_file}")
    print("OUT columns:", merged_df.columns.tolist()[:8], "...")
    print("OUT exp_sid null count:", merged_df["exp_sid"].isna().sum())

if __name__ == "__main__":
    input_folder = root / "DATA" / "2_lab"
    output_file = root / "ANALYSIS" / "1_assembled.csv"
    merge_csvs(str(input_folder), str(output_file))
