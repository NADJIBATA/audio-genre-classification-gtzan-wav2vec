#!/usr/bin/env python3
"""Properly format labels file for fairseq audio classification."""

# Create dict.labels.txt in proper fairseq format
with open('data/fairseq/dict.labels.txt', 'w') as f:
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    for genre in genres:
        f.write(f"{genre} 1\n")

print("✔ Created data/fairseq/dict.labels.txt")

# Verify the .labels files format (should be just integers, one per line)
for split in ['train', 'valid', 'test']:
    labels_file = f'data/fairseq/{split}.labels'
    # Read first few lines to verify format
    with open(labels_file, 'r') as f:
        lines = f.readlines()[:5]
    print(f"\n{split}.labels (first 5 lines):")
    for line in lines:
        print(f"  {line.strip()}")
    print(f"  Total: {len(open(labels_file).readlines())} labels")
