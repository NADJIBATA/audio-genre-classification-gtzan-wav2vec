import os

def fix_manifest_paths(manifest_file, audio_dir="data/fairseq/audio"):
    """
    Fix paths in fairseq manifest files to use absolute or correct relative paths.
    
    Args:
        manifest_file: Path to the manifest file (train.tsv, valid.tsv, etc.)
        audio_dir: Directory containing audio files
    """
    if not os.path.exists(manifest_file):
        print(f"Warning: {manifest_file} not found")
        return
    
    print(f"Fixing paths in {manifest_file}...")
    
    # Read the manifest
    with open(manifest_file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        print(f"Warning: {manifest_file} is empty")
        return
    
    # First line should be the root path
    # Replace it with the correct audio directory path
    audio_dir_abs = os.path.abspath(audio_dir)
    lines[0] = audio_dir_abs + '\n'
    
    # Write back the fixed manifest
    with open(manifest_file, 'w') as f:
        f.writelines(lines)
    
    print(f"✓ Fixed: First line now points to {audio_dir_abs}")
    print(f"  Total entries: {len(lines) - 1}")

def verify_audio_files(manifest_file, audio_dir="data/fairseq/audio"):
    """
    Verify that all audio files referenced in manifest actually exist.
    """
    if not os.path.exists(manifest_file):
        return
    
    print(f"\nVerifying audio files for {manifest_file}...")
    
    with open(manifest_file, 'r') as f:
        lines = f.readlines()
    
    # Skip first line (root path)
    missing_files = []
    for i, line in enumerate(lines[1:], 1):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            relative_path = parts[0]
            full_path = os.path.join(audio_dir, relative_path)
            
            if not os.path.exists(full_path):
                missing_files.append((i, relative_path, full_path))
    
    if missing_files:
        print(f"⚠ Found {len(missing_files)} missing files:")
        for idx, rel_path, full_path in missing_files[:5]:  # Show first 5
            print(f"  Line {idx}: {rel_path}")
            print(f"    Expected at: {full_path}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return False
    else:
        print(f"✓ All {len(lines) - 1} audio files found")
        return True

if __name__ == "__main__":
    # Auto-detect environment
    is_colab = "/content/drive" in os.getcwd() or "/content" in os.getcwd()
    
    if is_colab:
        base_path = "/content/drive/MyDrive/NADJIBATA-audio-genre-classification"
        data_dir = f"{base_path}/data/fairseq"
        audio_dir = f"{base_path}/data/fairseq/audio"
    else:
        data_dir = "data/fairseq"
        audio_dir = "data/fairseq/audio"
    
    print("=" * 60)
    print("FIXING FAIRSEQ MANIFEST PATHS")
    print("=" * 60)
    print(f"Environment: {'Colab' if is_colab else 'Local'}")
    print(f"Data directory: {data_dir}")
    print(f"Audio directory: {audio_dir}")
    print()
    
    # Fix all manifest files
    for split in ['train', 'valid', 'test']:
        manifest_file = os.path.join(data_dir, f"{split}.tsv")
        fix_manifest_paths(manifest_file, audio_dir)
        verify_audio_files(manifest_file, audio_dir)
        print()
    
    print("=" * 60)
    print("DONE! Now re-run your training script.")
    print("=" * 60)