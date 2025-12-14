#!/usr/bin/env python3
"""Regenerate labels file from the updated TSV files."""

import os
import numpy as np

# Read train.tsv to extract labels
TSV_FILE = 'data/fairseq/train.tsv'
LABELS_OUTPUT = 'data/fairseq/dict.labels.txt'

labels_set = set()
with open(TSV_FILE, 'r') as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            try:
                label = int(parts[1])
                labels_set.add(label)
            except ValueError:
                pass

# Get unique labels sorted
labels = sorted(list(labels_set))
print(f"Found {len(labels)} unique labels: {labels}")

# Write labels file in fairseq format (label_name label_id)
with open(LABELS_OUTPUT, 'w') as f:
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for idx, genre in enumerate(genres):
        if idx in labels:
            f.write(f"{genre} {idx}\n")

print(f"✔ Regenerated {LABELS_OUTPUT}")

# Count samples per split
for split_file in ['data/fairseq/train.tsv', 'data/fairseq/valid.tsv', 'data/fairseq/test.tsv']:
    count = 0
    with open(split_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            if line.strip():
                count += 1
    print(f"  {split_file}: {count} samples")
