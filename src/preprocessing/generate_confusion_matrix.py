
"""
Génération de la matrice de confusion avec prédictions détaillées
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import soundfile as sf

# Auto-detect environment
is_colab = "/content/drive" in os.getcwd() or "/content" in os.getcwd()

if is_colab:
    BASE_PATH = r"D:\NADJIBATA-audio-genre-classification"
    DATA_DIR = f"{BASE_PATH}/data/fairseq"
    CHECKPOINT_PATH = f"{BASE_PATH}/models/wav2vec/checkpoints/checkpoint9.pt"
    RESULTS_DIR = f"{BASE_PATH}/results"
else:
    DATA_DIR = "data/fairseq"
    CHECKPOINT_PATH = "models/wav2vec/checkpoints/checkpoint9.pt"
    RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("📊 GÉNÉRATION MATRICE DE CONFUSION")
print("="*70)

# 1. Charger le dictionnaire des labels
dict_path = os.path.join(DATA_DIR, "dict.labels.txt")
labels = []
with open(dict_path, 'r') as f:
    for line in f:
        if '<pad>' not in line and '<unk>' not in line:
            label = line.split()[0]
            labels.append(label)

print(f"✓ {len(labels)} classes: {', '.join(labels)}")

# 2. Charger le modèle
print(f"\n🔧 Chargement du modèle...")

try:
    # Utiliser fairseq pour charger le modèle
    from fairseq import checkpoint_utils

    # Load model and task together (handles checkpoint cfg formats)
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])

    # Charger le dataset de test
    task.load_dataset('test')

    model = models[0]
    
    # GPU/CPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        print(f"✓ Modèle sur GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Modèle sur CPU")
    
    model.eval()
    
    # 3. Faire les prédictions
    print(f"\n🔮 Prédictions sur le test set...")
    
    y_true = []
    y_pred = []
    filenames = []
    
    # Lire test.tsv pour avoir les noms de fichiers
    test_tsv = os.path.join(DATA_DIR, "test.tsv")
    with open(test_tsv, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    test_data = task.dataset('test')
    
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            # Préparer l'input
            source = sample['source'].unsqueeze(0)

            # récupérer la vraie étiquette (plusieurs clés possibles selon la version de fairseq)
            target = None
            for k in ('target', 'label', 'labels'):
                if k in sample:
                    target = sample[k]
                    break
            if target is None:
                # fallback: some datasets store target under 'net_input' or inside a tuple
                # try to read label from dataset's labels file if available
                raise KeyError('target')
            
            if use_cuda:
                source = source.cuda()

            # Prédiction
            net_output = model(source=source, padding_mask=None)
            logits = net_output[0]
            pred = logits.argmax(dim=-1).cpu().item()

            # target may be a tensor; try to get integer value
            try:
                true_idx = int(target.item()) if hasattr(target, 'item') else int(target)
            except Exception:
                true_idx = int(target[0]) if isinstance(target, (list, tuple)) else int(target)

            y_true.append(true_idx)
            y_pred.append(pred)
            
            # Extraire le nom du fichier
            if i < len(lines):
                filename = lines[i].split('\t')[0]
                filenames.append(filename)
            
            if (i + 1) % 10 == 0:
                print(f"  Traité: {i+1}/{len(test_data)}", end='\r')
    
    print(f"\n✓ {len(y_true)} échantillons traités")
    
    # 4. Générer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. Visualiser
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Nombre de prédictions'})
    plt.title('Matrice de Confusion - Test Set', fontsize=16, pad=20)
    plt.ylabel('Vraie classe', fontsize=12)
    plt.xlabel('Classe prédite', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Sauvegarder
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Matrice sauvegardée: {cm_path}")
    
    # 6. Rapport de classification
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print("\n" + "="*70)
    print("📈 RAPPORT DE CLASSIFICATION")
    print("="*70)
    print(report)
    
    # Sauvegarder le rapport
    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Rapport sauvegardé: {report_path}")
    
    # 7. Analyse des erreurs
    print("\n" + "="*70)
    print("🔍 ANALYSE DES ERREURS")
    print("="*70)
    
    errors = []
    for i, (true_label, pred_label, filename) in enumerate(zip(y_true, y_pred, filenames)):
        if true_label != pred_label:
            errors.append({
                'filename': filename,
                'true': labels[true_label],
                'predicted': labels[pred_label]
            })
    
    print(f"\nNombre d'erreurs: {len(errors)} / {len(y_true)} ({len(errors)/len(y_true)*100:.1f}%)")
    print(f"Accuracy: {(len(y_true)-len(errors))/len(y_true)*100:.2f}%")
    
    if errors:
        print("\n🔴 Top 10 erreurs:")
        for i, err in enumerate(errors[:10], 1):
            print(f"{i:2d}. {err['filename']:40s} | Vrai: {err['true']:12s} | Prédit: {err['predicted']}")
        
        # Sauvegarder toutes les erreurs
        errors_path = os.path.join(RESULTS_DIR, "errors_detail.txt")
        with open(errors_path, 'w') as f:
            f.write("ERREURS DE CLASSIFICATION\n")
            f.write("="*80 + "\n\n")
            for err in errors:
                f.write(f"{err['filename']}\n")
                f.write(f"  Vraie classe:  {err['true']}\n")
                f.write(f"  Classe prédite: {err['predicted']}\n\n")
        print(f"\n✓ Détail des erreurs: {errors_path}")
    
    # 8. Accuracy par classe
    print("\n" + "="*70)
    print("📊 ACCURACY PAR CLASSE")
    print("="*70)
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        class_total[true_label] += 1
        if true_label == pred_label:
            class_correct[true_label] += 1
    
    for label_idx in sorted(class_total.keys()):
        acc = class_correct[label_idx] / class_total[label_idx] * 100
        print(f"{labels[label_idx]:12s} : {acc:6.2f}% ({class_correct[label_idx]:3d}/{class_total[label_idx]:3d})")
    
    print("\n" + "="*70)
    print("✅ ÉVALUATION COMPLÈTE TERMINÉE")
    print("="*70)
    print(f"\n📁 Tous les résultats dans: {RESULTS_DIR}/")
    
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)