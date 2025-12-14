import os
import json
import glob
import random
import shutil

# Resolve paths relative to the project root (one level above this src file)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

RAW_DATA = os.path.join(REPO_ROOT, "data", "raw", "Data", "genres_original")
OUT_DIR = os.path.join(REPO_ROOT, "data", "fairseq")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/audio", exist_ok=True)

labels = sorted(os.listdir(RAW_DATA))
with open(f"{OUT_DIR}/labels.json", "w") as f:
    json.dump({"labels": labels}, f)

# write a simple fairseq dictionary file expected by audio_classification
# format: one token and a count per line (no special symbols)
dict_path = os.path.join(OUT_DIR, "dict.labels.txt")
with open(dict_path, "w", encoding="utf-8") as df:
    for lbl in labels:
        df.write(f"{lbl} 1\n")

all_files = []
for label in labels:
    files = glob.glob(f"{RAW_DATA}/{label}/*.wav")
    for fpath in files:
        fname = label + "_" + os.path.basename(fpath)
        new_path = f"{OUT_DIR}/audio/{fname}"
        try:
            # Skip if already exists
            if not os.path.exists(new_path):
                shutil.copy(fpath, new_path)
            all_files.append((fname, labels.index(label)))
        except (OSError, IOError, PermissionError) as e:
            print(f"⚠ Skipping {fpath}: {e}")
            continue

random.shuffle(all_files)
n = len(all_files)

splits = {
    "train": all_files[:int(0.8*n)],
    "valid": all_files[int(0.8*n):int(0.9*n)],
    "test":  all_files[int(0.9*n):]
}

for split, items in splits.items():
    # write manifest TSV
    with open(f"{OUT_DIR}/{split}.tsv", "w") as f:
        f.write(os.path.abspath(f"{OUT_DIR}/audio") + "\n")
        for fname, label in items:
            f.write(f"{fname}\t{label}\n")

    # write label file (one text label per line) for Fairseq audio_classification
    labels_file = os.path.join(OUT_DIR, f"{split}.labels")
    with open(labels_file, "w", encoding="utf-8") as lf:
        for fname, label in items:
            lf.write(f"{labels[label]}\n")

print("✔ TSV et labels.json générés !")
