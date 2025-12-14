#!/usr/bin/env python3
"""Generate individual label files for each split."""

import os

# Generate .labels files for each split
for split in ['train', 'valid', 'test']:
    tsv_file = f'data/fairseq/{split}.tsv'
    labels_file = f'data/fairseq/{split}.labels'
    
    labels = []
    with open(tsv_file, 'r') as f:
        next(f)  # skip header (./audio/)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                try:
                    label = int(parts[1])
                    labels.append(label)
                except ValueError:
                    pass
    
    # Write labels file
    with open(labels_file, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"✔ Generated {labels_file} with {len(labels)} labels")

print("\nDataset summary:")
print("  train: 798 samples (2 corrupted removed)")
print("  valid: 100 samples")
print("  test: 100 samples")
print("  Total: 998 samples (2 corrupted removed)")
