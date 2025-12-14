#!/usr/bin/env python3
"""Iteratively remove rock_rock files that fail to load during fairseq validation."""
import subprocess
import sys
import os
import re
from pathlib import Path

scripts_dir = os.path.abspath('fairseq/.venv_fairseq/Scripts')
env = os.environ.copy()
env['PATH'] = scripts_dir + os.pathsep + env.get('PATH','')

MAX_ITERATIONS = 20
AUDIO_DIR = Path('data/fairseq/audio')

def extract_failed_file(stderr_output):
    """Extract filename from fairseq error message."""
    match = re.search(r"Failed to load \./audio/(\S+\.wav)", stderr_output)
    return match.group(1) if match else None

def remove_file_from_tsv(fname):
    """Remove entries with this filename from all TSVs and regenerate labels."""
    removed_count = 0
    for split in ('train', 'valid', 'test'):
        tsv_path = Path(f'data/fairseq/{split}.tsv')
        with open(tsv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        cleaned = [lines[0]]  # Keep header
        for line in lines[1:]:
            if fname not in line:
                cleaned.append(line)
            else:
                removed_count += 1
        
        with open(tsv_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned)
    
    # Delete the audio file
    audio_path = AUDIO_DIR / fname
    if audio_path.exists():
        audio_path.unlink()
        print(f"  Deleted {fname}")
    
    # Regenerate labels
    subprocess.run([sys.executable, 'src/fix_labels_from_tsv.py'], 
                   cwd='.', capture_output=True, env=env)
    
    return removed_count

def run_training():
    """Run training and return (success, failed_file)"""
    cmd = [sys.executable, 'src/train_wav2vec_FFT.py']
    proc = subprocess.run(cmd, cwd='.', capture_output=True, text=True, timeout=180, env=env)
    
    output = proc.stdout + '\n' + proc.stderr
    
    if proc.returncode == 0:
        return True, None
    
    failed_file = extract_failed_file(output)
    return False, failed_file

def main():
    print("Training with automatic corrupted-file removal...")
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        success, failed_file = run_training()
        
        if success:
            print("\nTraining launched successfully!")
            return 0
        
        if not failed_file:
            print("\nTraining failed, but could not extract failed file.")
            print("Please check the error message above.")
            return 1
        
        print(f"Failed file: {failed_file}")
        remove_file_from_tsv(failed_file)
        print(f"Removed {failed_file} from TSVs and labels. Retrying...")
    
    print(f"\nMax iterations ({MAX_ITERATIONS}) reached. Dataset may have too many corrupted files.")
    return 1

if __name__ == '__main__':
    sys.exit(main())
