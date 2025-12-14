import os
import torch

DATA_DIR = "data/fairseq"
W2V_PATH = "models/wav2vec/wav2vec_small.pt"
SAVE_DIR = "models/wav2vec/checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

use_cuda = torch.cuda.is_available()
print(f"CUDA available: {use_cuda}")
if use_cuda:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

args = [
    f"fairseq-train {DATA_DIR}",
    "--task audio_classification",
    "--arch wav2vec_classification",
    f"--w2v-path {W2V_PATH}",
    "--labels labels",
    "--sample-rate 22050",
    "--pooling mean",

    "--max-update 2000",
    "--lr 1e-3",
    "--warmup-updates 500",
    "--max-sample-size 660000",
    "--max-tokens 660000",
    "--optimizer adam",
    "--adam-eps 1e-06",
    "--weight-decay 0.0",
    "--criterion cross_entropy",

    "--best-checkpoint-metric accuracy",
    "--maximize-best-checkpoint-metric",   

    "--batch-size 4",
    "--log-format simple",
    "--log-interval 20",
    f"--save-dir {SAVE_DIR}",

    "--freeze-finetune-updates 999999",
    "--feature-grad-mult 0.0",
    "--patience 10",
]


TRAIN_HEAD_ONLY = True

if TRAIN_HEAD_ONLY:
    print("\n>>> Mode: Head-only training enabled — pretrained backbone will remain frozen")
else:
    print("\n>>> Mode: Full fine-tuning (backbone trainable)")

if use_cuda:
    args.append("--fp16")   
else:
    args.append("--cpu")

cmd = " ".join(args)

print(f"\nRunning: {cmd}\n")
os.system(cmd)