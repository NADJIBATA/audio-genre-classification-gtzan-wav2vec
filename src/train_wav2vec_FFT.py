import os
import torch

is_colab = "/content/drive" in os.getcwd() or "/content" in os.getcwd()

if is_colab:
    BASE_PATH = "/content/drive/MyDrive/NADJIBATA-audio-genre-classification"
    DATA_DIR = f"{BASE_PATH}/data/fairseq"
    W2V_PATH = f"{BASE_PATH}/models/wav2vec/wav2vec_small.pt"
    SAVE_DIR = f"{BASE_PATH}/models/wav2vec/checkpoints"
else:
    DATA_DIR = "data/fairseq"
    W2V_PATH = "models/wav2vec/wav2vec_small.pt"
    SAVE_DIR = "models/wav2vec/checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

use_cuda = torch.cuda.is_available()
print(f"Environment: {'Colab' if is_colab else 'Local'}")
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
    "--max-epoch 12",
    "--max-update 20000",
    "--lr 1e-5",
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
]

if is_colab :
    args.append("--fp16") 
elif use_cuda:
    args.append("--fp16")  
else:
    args.append("--cpu")   

cmd = " ".join(args)

print(f"\nRunning: {cmd}\n")
os.system(cmd)
