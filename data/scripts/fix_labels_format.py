#!/usr/bin/env python3
"""Convert numeric labels to genre names for fairseq audio classification."""

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for split in ['train', 'valid', 'test']:
    tsv_file = f'data/fairseq/{split}.tsv'
    labels_file = f'data/fairseq/{split}.labels'
    
    # Read TSV and convert numeric labels to genre names
    labels = []
    with open(tsv_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                try:
                    label_idx = int(parts[1])
                    label_name = genres[label_idx]
                    labels.append(label_name)
                except (ValueError, IndexError) as e:
                    print(f"Error processing line in {split}.tsv: {line} - {e}")
    
    # Write labels file with genre names
    with open(labels_file, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    
    print(f"✔ Converted {labels_file} ({len(labels)} labels)")
    with open(labels_file, 'r') as f:
        first_5 = [f.readline().strip() for _ in range(5)]
    print(f"  First 5: {first_5}")
