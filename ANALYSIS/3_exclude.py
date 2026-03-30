#!/usr/bin/python3
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "ANALYSIS" / "2_simplified.csv"
OUTFILE = ROOT / "ANALYSIS" / "3_purified.csv"


def exclude_low_language_engagement(df):
    mean_words = pd.to_numeric(df.get("mean_words_per_turn"), errors="coerce")
    word_rate = pd.to_numeric(df.get("word_rate_wps"), errors="coerce")

    return (mean_words < 10) & (word_rate < 1.5)


def main():
    df = pd.read_csv(INFILE)

    exclusion_mask = exclude_low_language_engagement(df)
    purified = df.loc[~exclusion_mask].copy()

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    purified.to_csv(OUTFILE, index=False)

    print("Input rows: {}".format(len(df)))
    print("Excluded rows: {}".format(int(exclusion_mask.sum())))
    print("Output rows: {}".format(len(purified)))
    print("Wrote {}".format(OUTFILE))


if __name__ == "__main__":
    main()
