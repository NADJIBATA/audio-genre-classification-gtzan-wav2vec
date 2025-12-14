#!/usr/bin/env python3
"""Fix absolute paths in fairseq TSV files to use relative paths"""

import os

tsv_files = [
    'data/fairseq/train.tsv',
    'data/fairseq/valid.tsv', 
    'data/fairseq/test.tsv'
]

# Old absolute path
old_path = r"C:\Users\anadj\Documents\audio-genre-classification-wav2vec\NADJIBATA-audio-genre-classification\data\fairseq\audio"
new_path = "./audio"

for tsv_file in tsv_files:
    if not os.path.exists(tsv_file):
        print(f"Skipping {tsv_file} - not found")
        continue
    
    with open(tsv_file, 'r') as f:
        lines = f.readlines()
    
    # Replace the first line (header with path)
    if lines and old_path in lines[0]:
        lines[0] = new_path + '\n'
        with open(tsv_file, 'w') as f:
            f.writelines(lines)
        print(f"Fixed {tsv_file}")
    else:
        print(f"Path not found in {tsv_file} or already fixed")

print("Done!")
