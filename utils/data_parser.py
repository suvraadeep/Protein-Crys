"""
Merges salt-conc and temperature datasets; parses pH and PEG type
from the raw REMARK 280 crystallization condition text.
"""

import re
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SALT_CSV = ROOT / "salt-conc" / "crystallization_dataset.csv"
TEMP_CSV = ROOT / "temp" / "metadata(1).csv"

# --- Canonical PEG class mapping ---
PEG_CLASSES = ["PEG_400", "PEG_1000", "PEG_1500", "PEG_2000",
               "PEG_3350", "PEG_4000", "PEG_6000", "PEG_8000", "OTHER_PEG"]

def _peg_mw_to_class(mw: int) -> str:
    boundaries = [600, 1250, 1750, 2500, 3700, 5000, 7000, 9000]
    labels = PEG_CLASSES
    for i, b in enumerate(boundaries):
        if mw <= b:
            return labels[i]
    return "OTHER_PEG"

# pH patterns — cover common formatting variants in PDB REMARK 280
_PH_PATTERNS = [
    re.compile(r'\bPH\s*[=:]?\s*(\d+\.?\d*)', re.IGNORECASE),
    re.compile(r'PH\s+VALUE\s*[=:]?\s*(\d+\.?\d*)', re.IGNORECASE),
    re.compile(r'AT\s+PH\s*(\d+\.?\d*)', re.IGNORECASE),
]

# PEG patterns
_PEG_PATTERNS = [
    re.compile(r'POLYETHYLENE\s+GLYCOL\s+(\d+)', re.IGNORECASE),
    re.compile(r'\bPEG\s*-?\s*(\d+)\b', re.IGNORECASE),
    re.compile(r'\bPEG(\d+)\b', re.IGNORECASE),
]


def parse_pH(remark_text: str) -> float | None:
    if not isinstance(remark_text, str):
        return None
    for pat in _PH_PATTERNS:
        m = pat.search(remark_text)
        if m:
            val = float(m.group(1))
            if 2.0 <= val <= 12.0:  # sanity bounds
                return val
    return None


def parse_peg_class(remark_text: str) -> str | None:
    if not isinstance(remark_text, str):
        return None
    for pat in _PEG_PATTERNS:
        m = pat.search(remark_text)
        if m:
            mw = int(m.group(1))
            if 100 <= mw <= 20000:
                return _peg_mw_to_class(mw)
    return None


def load_and_merge_datasets(
    salt_csv: str | Path = SALT_CSV,
    temp_csv: str | Path = TEMP_CSV,
) -> pd.DataFrame:
    """
    Returns a unified DataFrame with columns:
      pdb_id, sequence, seq_length,
      pH (float, NaN if missing),
      salt_concentration_M (float, NaN if missing),
      salt_type (str, NaN if missing),
      peg_class (str, NaN if missing),
      temp_k (float, NaN if missing)
    """
    salt_df = pd.read_csv(salt_csv)
    temp_df = pd.read_csv(temp_csv)

    # Normalise column names
    salt_df.columns = salt_df.columns.str.strip()
    temp_df.columns = temp_df.columns.str.strip()

    # Deduplicate on pdb_id within each source
    salt_df = salt_df.drop_duplicates(subset="pdb_id")
    temp_df = temp_df.drop_duplicates(subset="pdb_id")

    # Parse pH and PEG from remark_280
    if "remark_280" in salt_df.columns:
        salt_df["pH"] = salt_df["remark_280"].apply(parse_pH)
        salt_df["peg_class"] = salt_df["remark_280"].apply(parse_peg_class)
    else:
        salt_df["pH"] = np.nan
        salt_df["peg_class"] = np.nan

    # Merge temperature in
    merged = salt_df.merge(
        temp_df[["pdb_id", "temp_k"]],
        on="pdb_id",
        how="left",
    )

    # Also bring in sequences from temp that have NO salt record
    temp_only = temp_df[~temp_df["pdb_id"].isin(salt_df["pdb_id"])].copy()
    temp_only["pH"] = np.nan
    temp_only["peg_class"] = np.nan
    temp_only["salt_concentration_M"] = np.nan
    temp_only["salt_type"] = np.nan
    if "seq_length" not in temp_only.columns:
        temp_only["seq_length"] = temp_only["sequence"].str.len()

    merged = pd.concat([merged, temp_only], ignore_index=True)

    # Ensure seq_length column
    if "seq_length" not in merged.columns:
        merged["seq_length"] = merged["sequence"].str.len()

    # Filter: keep only standard amino-acid sequences, length 30-1500
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    def _is_valid(seq):
        if not isinstance(seq, str):
            return False
        s = seq.upper().replace(" ", "")
        return 30 <= len(s) <= 1500 and all(c in valid_aa for c in s)

    merged = merged[merged["sequence"].apply(_is_valid)].copy()
    merged["sequence"] = merged["sequence"].str.upper().str.replace(" ", "")
    merged["seq_length"] = merged["sequence"].str.len()

    merged = merged.reset_index(drop=True)
    return merged


def get_peg_label_mapping(df: pd.DataFrame) -> dict[str, int]:
    """Returns {peg_class_str: int_label} for classes present in df."""
    present = sorted(df["peg_class"].dropna().unique().tolist())
    return {cls: i for i, cls in enumerate(present)}


def print_dataset_stats(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"Unified dataset: {len(df):,} total rows")
    print(f"{'='*55}")
    for col, label in [
        ("pH", "pH"),
        ("salt_concentration_M", "Salt Conc (M)"),
        ("peg_class", "PEG Type"),
        ("temp_k", "Temperature (K)"),
    ]:
        n = df[col].notna().sum()
        pct = n / len(df) * 100
        print(f"  {label:<22}: {n:5,} rows  ({pct:.1f}%)")
    if "peg_class" in df.columns:
        print(f"\n  PEG class distribution:")
        for cls, cnt in df["peg_class"].value_counts().items():
            print(f"    {cls:<15}: {cnt:,}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    df = load_and_merge_datasets()
    print_dataset_stats(df)
    mapping = get_peg_label_mapping(df)
    print("PEG label mapping:", mapping)
