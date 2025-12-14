#!/usr/bin/env python3
"""Detect and remove corrupted entries from fairseq TSVs by comparing audio file availability to TSV."""
import os
from pathlib import Path

ROOT = Path('data/fairseq')
AUDIO_DIR = ROOT / 'audio'

def check_and_clean_strict(split):
    """Remove any TSV entry whose audio file doesn't exist in ./audio/ directory."""
    tsv_path = ROOT / f"{split}.tsv"
    if not tsv_path.exists():
        print(f"Skipping {split}: {tsv_path} not found")
        return 0

    # List all existing audio files
    audio_files = set(f.name for f in AUDIO_DIR.glob('*.wav'))

    lines = []
    removed = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 2:
                removed.append((line.strip(), 'malformed'))
                continue
            
            fname = parts[0]
            # Handle both relative paths like 'rock_rock.00085.wav' and './audio/rock_rock.00085.wav'
            if fname.startswith('./audio/') or fname.startswith('.\\audio\\'):
                fname = fname.split('/')[-1].split('\\')[-1]
            
            if fname not in audio_files:
                removed.append((line.strip(), 'file_not_found'))
            else:
                lines.append(line)

    if removed:
        # Rewrite TSV
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.write(header)
            for l in lines:
                f.write(l)
        
        # Log removed entries
        with open(ROOT / 'corrupted_files.txt', 'a', encoding='utf-8') as cf:
            for entry, reason in removed:
                cf.write(f"{split}\t{entry}\t{reason}\n")

    return len(removed)

def main():
    # Clear previous log
    log_path = ROOT / 'corrupted_files.txt'
    if log_path.exists():
        log_path.unlink()
    
    total_removed = 0
    for split in ('train', 'valid', 'test'):
        print(f"Checking {split}...", end=' ')
        removed = check_and_clean_strict(split)
        print(f"removed {removed} entries")
        total_removed += removed

    if total_removed > 0:
        print(f"{total_removed} entries removed. Regenerating .labels files...")
        # regenerate labels from tsv
        from fix_labels_from_tsv import main as regen_labels
        regen_labels()
        print("Labels regenerated.")
    else:
        print("All files present in dataset.")

if __name__ == '__main__':
    main()
