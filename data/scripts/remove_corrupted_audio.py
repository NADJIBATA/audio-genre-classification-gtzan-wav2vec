#!/usr/bin/env python3
"""Remove corrupted audio files and their entries from TSV files."""

import os
import soundfile as sf

AUDIO_DIR = os.path.join('data', 'fairseq', 'audio')
TSV_FILES = [
    'data/fairseq/train.tsv',
    'data/fairseq/valid.tsv',
    'data/fairseq/test.tsv'
]

# Find all corrupted files
corrupted = []
all_files = sorted(os.listdir(AUDIO_DIR))
print(f"Checking {len(all_files)} audio files...")

for fname in all_files:
    fpath = os.path.join(AUDIO_DIR, fname)
    try:
        info = sf.info(fpath)
        # Try to read a small portion to verify it's actually readable
        data, sr = sf.read(fpath, frames=100)
    except Exception as e:
        print(f"✗ Corrupted: {fname} - {e}")
        corrupted.append(fname)

if not corrupted:
    print("✔ No corrupted files found!")
else:
    print(f"\n✔ Found {len(corrupted)} corrupted files. Removing from TSVs...")
    
    # Remove corrupted entries from all TSV files
    for tsv_file in TSV_FILES:
        if not os.path.exists(tsv_file):
            print(f"  Skipping {tsv_file} - not found")
            continue
        
        with open(tsv_file, 'r') as f:
            lines = f.readlines()
        
        # Keep header (first line) and filter out lines with corrupted files
        new_lines = [lines[0]]  # header
        removed_count = 0
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 1 and parts[0] not in corrupted:
                new_lines.append(line)
            else:
                removed_count += 1
        
        with open(tsv_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"  {tsv_file}: removed {removed_count} entries")

print("\n✔ Done! Corrupted files removed from dataset.")
