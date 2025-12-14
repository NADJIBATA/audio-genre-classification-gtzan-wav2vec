"""
Script d'évaluation complète du modèle Wav2Vec2 sur le test set
Génère métriques détaillées + matrice de confusion
"""

import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
import json

# Auto-detect environment
is_colab = "/content/drive" in os.getcwd() or "/content" in os.getcwd()

if is_colab:
    BASE_PATH = r"D:\NADJIBATA-audio-genre-classification"
    DATA_DIR = f"{BASE_PATH}/data/fairseq"
    CHECKPOINT_PATH = f"{BASE_PATH}/models/wav2vec/checkpoints_head_only/checkpoint12.pt"
    RESULTS_DIR = f"{BASE_PATH}/results"
else:
    DATA_DIR = "data/fairseq"
    CHECKPOINT_PATH = "models/wav2vec/checkpoints_head_only/checkpoint12.pt"
    RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Vérifier que le checkpoint existe
if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint introuvable: {CHECKPOINT_PATH}")
    print("\nCheckpoints diMyDrivesponibles:")
    checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
    if os.path.exists(checkpoint_dir):
        for f in sorted(os.listdir(checkpoint_dir)):
            if f.endswith('.pt'):
                print(f"  - {f}")
    sys.exit(1)

print("="*70)
print("📊 ÉVALUATION DU MODÈLE WAV2VEC2 SUR TEST SET")
print("="*70)

# Construire la commande fairseq-validate
cmd_parts = [
    "fairseq-validate",
    DATA_DIR,
    "--task", "audio_classification",
    "--path", CHECKPOINT_PATH,
    "--valid-subset", "test",  # ← Utiliser le test set
    "--batch-size", "8",
    "--results-path", RESULTS_DIR,
]

# Ajouter GPU/CPU
use_cuda = torch.cuda.is_available()
if use_cuda:
    # Use GPU. Prefer FP32 to avoid fp16/FP32 mismatches with some checkpoints.
    # Enable FP16 only when explicitly requested via env var `USE_FP16=1`.
    if os.environ.get("USE_FP16") == "1":
        cmd_parts.append("--fp16")
        print(f"✓ GPU détecté: {torch.cuda.get_device_name(0)} (fp16 enabled)")
    else:
        print(f"✓ GPU détecté: {torch.cuda.get_device_name(0)} (running in FP32 to avoid precision mismatch). Set USE_FP16=1 to enable fp16")
else:
    cmd_parts.append("--cpu")
    print("⚠️  CPU mode")

print(f"✓ Checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
print(f"✓ Test set: {DATA_DIR}/test.tsv")
print("="*70)

cmd = " ".join(cmd_parts)
print(f"\n🚀 Commande: {cmd}\n")

# Exécuter l'évaluation en capturant la sortie
import subprocess
import datetime

proc = subprocess.run(cmd_parts, capture_output=True, text=True)
stdout = proc.stdout or ""
stderr = proc.stderr or ""
exit_code = proc.returncode

if exit_code == 0:
    print("\n" + "="*70)
    print("✅ ÉVALUATION TERMINÉE AVEC SUCCÈS")
    print("="*70)

    # Tenter d'extraire loss et accuracy depuis la sortie
    test_loss = None
    test_accuracy = None
    # Chercher les lignes contenant "valid" et "test" subset
    for line in stdout.splitlines():
        line = line.strip()
        m = None
        # Exemple de ligne: "valid | epoch 001 | valid on 'valid' subset | loss 3.141 | ppl 8.82 | accuracy 0.322917 | ..."
        if "test' subset" in line or "valid on 'test' subset" in line or "valid on \"test\" subset" in line:
            m = re.search(r"loss\s+([0-9\.]+).*accuracy\s+([0-9\.]+)", line)
        else:
            # fallback: chercher toute ligne qui contient 'valid' and 'accuracy' and 'test' nearby
            if 'test' in line and 'accuracy' in line:
                m = re.search(r"loss\s+([0-9\.]+).*accuracy\s+([0-9\.]+)", line)
        if m:
            try:
                test_loss = float(m.group(1))
                acc_raw = float(m.group(2))
                test_accuracy = acc_raw * 100.0 if acc_raw <= 1.0 else acc_raw
                break
            except Exception:
                continue

    # Si on n'a pas trouvé dans stdout, essayer d'inspecter des fichiers résultats (fairseq peut écrire des .txt)
    if test_accuracy is None:
        for p in Path(RESULTS_DIR).glob("*.txt"):
            try:
                txt = p.read_text(encoding='utf-8')
                m = re.search(r"accuracy\s+([0-9\.]+)", txt)
                if m:
                    acc_raw = float(m.group(1))
                    test_accuracy = acc_raw * 100.0 if acc_raw <= 1.0 else acc_raw
                m2 = re.search(r"loss\s+([0-9\.]+)", txt)
                if m2:
                    test_loss = float(m2.group(1))
                if test_accuracy is not None:
                    break
            except Exception:
                continue

    # Afficher et sauvegarder les résultats
    if test_accuracy is not None:
        print(f"\n📁 Test metrics: accuracy={test_accuracy:.4f}%, loss={test_loss if test_loss is not None else 'N/A'}")
    else:
        print("\n⚠️  Impossible d'extraire l'accuracy du test depuis la sortie; vérifier manuellement les fichiers dans results/.")

    # Lister les fichiers produits
    result_files = list(Path(RESULTS_DIR).glob("*.txt"))
    if result_files:
        print(f"\n📄 Résultats sauvegardés dans: {RESULTS_DIR}")
        for f in result_files:
            print(f"  - {f.name}")

    # Sauvegarder test metrics de façon persistante
    metrics = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'checkpoint': os.path.basename(CHECKPOINT_PATH),
        'test_loss': test_loss,
        'test_accuracy_percent': test_accuracy
    }
    metrics_path = os.path.join(RESULTS_DIR, 'test_metrics.json')
    try:
        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=2)
        print(f"\n✓ Test metrics sauvegardés: {metrics_path}")
    except Exception as e:
        print(f"⚠️  Impossible d'écrire {metrics_path}: {e}")

else:
    print("\n❌ Erreur lors de l'évaluation")
    print(stderr)
    sys.exit(1)

print("\n💡 Prochaine étape: Générez la matrice de confusion avec:")
print("   python generate_confusion_matrix.py")