#!/usr/bin/env python3
"""Regenerate per-split .labels files from data/fairseq/*.tsv and dict.labels.txt

This ensures the number of label lines matches the number of samples in each TSV.
"""
import os
from pathlib import Path

DATA_DIR = Path("data/fairseq")
DICT_PATH = DATA_DIR / "dict.labels.txt"

def load_label_map():
    # dict.labels.txt lines: <label> <id>
    mapping = {}
    if not DICT_PATH.exists():
        raise FileNotFoundError(f"{DICT_PATH} not found")
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            label = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                mapping[int(parts[1])] = label
            else:
                mapping[len(mapping)] = label
    return mapping

def fix_split(split):
    tsv = DATA_DIR / f"{split}.tsv"
    out_labels = DATA_DIR / f"{split}.labels"
    if not tsv.exists():
        print(f"Skipping {split}: {tsv} not found")
        return

    mapping = load_label_map()

    with open(tsv, 'r', encoding='utf-8') as f_in, open(out_labels, 'w', encoding='utf-8') as f_out:
        header = f_in.readline()
        count = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                # malformed line — skip
                print(f"Malformed line in {tsv}: {line}")
                continue
            # parts[1] may be a numeric id (0-based or 1-based) or a string label
            raw_label = parts[1]
            # if it's already a non-numeric label, write as-is
            try:
                label_id = int(raw_label)
            except ValueError:
                f_out.write(raw_label + "\n")
                count += 1
                continue

            # Try mapping with the id as-is, then try common offsets if missing.
            label_name = mapping.get(label_id)
            if label_name is None:
                # try 1-based vs 0-based mismatch
                label_name = mapping.get(label_id + 1)
            if label_name is None:
                label_name = mapping.get(label_id - 1)
            if label_name is None:
                # fallback to string of id
                label_name = str(label_id)

            f_out.write(f"{label_name}\n")
            count += 1

    print(f"Wrote {count} labels to {out_labels}")

def main():
    for split in ("train", "valid", "test"):
        fix_split(split)

if __name__ == '__main__':
    main()
