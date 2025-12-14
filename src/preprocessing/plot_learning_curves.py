
"""
Visualisation des courbes d'apprentissage depuis les logs
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Auto-detect environment
is_colab = "/content/drive" in os.getcwd() or "/content" in os.getcwd()

if is_colab:
    BASE_PATH = "/content/drive/MyDrive/NADJIBATA-audio-genre-classification"
    RESULTS_DIR = f"{BASE_PATH}/results"
else:
    RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("📈 VISUALISATION DES COURBES D'APPRENTISSAGE")
print("="*70)

# Données extraites des nouveaux logs (30 secondes d'audio)
epochs = list(range(1, 13))

train_loss = [3.297, 2.894, 2.426, 1.883, 1.416, 1.244, 0.912, 0.778, 0.585, 0.438, 0.314, 0.247]
train_acc = [15.6, 35.6, 43.8, 56.4, 68.1, 70.9, 78.4, 83.4, 87.3, 91.3, 93.6, 95.4]

valid_loss = [3.081, 2.644, 2.298, 1.675, 1.639, 1.295, 1.244, 1.125, 1.067, 1.173, 1.037, 0.906]
valid_acc = [36.5, 45.8, 49.0, 62.5, 57.3, 66.7, 67.7, 70.8, 74.0, 76.0, 78.1, 81.3]

print(f"✓ Données chargées: {len(epochs)} epochs")

# Créer la figure avec 2 sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ===== GRAPHIQUE 1: LOSS =====
ax1.plot(epochs, train_loss, 'o-', linewidth=2, markersize=8, 
         label='Train Loss', color='#2E86AB', alpha=0.8)
ax1.plot(epochs, valid_loss, 's-', linewidth=2, markersize=8, 
         label='Valid Loss', color='#A23B72', alpha=0.8)

# Marquer le meilleur epoch
best_epoch = epochs[np.argmin(valid_loss)]
best_valid_loss = min(valid_loss)
ax1.plot(best_epoch, best_valid_loss, '*', markersize=20, 
         color='gold', markeredgecolor='black', markeredgewidth=2,
         label=f'Best (Epoch {best_epoch})')

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Évolution de la Loss', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, max(epochs) + 1)

# Annoter le meilleur point
ax1.annotate(f'{best_valid_loss:.2f}', 
             xy=(best_epoch, best_valid_loss),
             xytext=(best_epoch-1, best_valid_loss+0.3),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

# ===== GRAPHIQUE 2: ACCURACY =====
ax2.plot(epochs, train_acc, 'o-', linewidth=2, markersize=8, 
         label='Train Accuracy', color='#2E86AB', alpha=0.8)
ax2.plot(epochs, valid_acc, 's-', linewidth=2, markersize=8, 
         label='Valid Accuracy', color='#A23B72', alpha=0.8)

# Marquer le meilleur epoch
best_epoch_acc = epochs[np.argmax(valid_acc)]
best_valid_acc = max(valid_acc)
ax2.plot(best_epoch_acc, best_valid_acc, '*', markersize=20, 
         color='gold', markeredgecolor='black', markeredgewidth=2,
         label=f'Best (Epoch {best_epoch_acc})')

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(0, max(epochs) + 1)
ax2.set_ylim(0, 100)

# Annoter le meilleur point
ax2.annotate(f'{best_valid_acc:.1f}%', 
             xy=(best_epoch_acc, best_valid_acc),
             xytext=(best_epoch_acc-2, best_valid_acc-8),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Ajouter zone de surapprentissage
for epoch, t_acc, v_acc in zip(epochs, train_acc, valid_acc):
    gap = t_acc - v_acc
    if gap > 15:  # Écart > 15%
        ax2.axvspan(epoch-0.5, epoch+0.5, alpha=0.15, color='red')

plt.tight_layout()

# Sauvegarder
output_path = os.path.join(RESULTS_DIR, "learning_curves.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Graphique sauvegardé: {output_path}")

# ===== GRAPHIQUE 3: ÉCART TRAIN/VALID =====
fig2, ax3 = plt.subplots(figsize=(12, 6))

gap = [t - v for t, v in zip(train_acc, valid_acc)]

colors = ['green' if g < 10 else 'orange' if g < 20 else 'red' for g in gap]
bars = ax3.bar(epochs, gap, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax3.axhline(y=10, color='orange', linestyle='--', linewidth=2, label='Seuil attention (10%)')
ax3.axhline(y=20, color='red', linestyle='--', linewidth=2, label='Seuil danger (20%)')

ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Écart Train - Valid (%)', fontsize=12, fontweight='bold')
ax3.set_title('Analyse du Surapprentissage (Overfitting)', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.set_xlim(0, max(epochs) + 1)

# Annoter les valeurs
for bar, g in zip(bars, gap):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{g:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()

# Sauvegarder
gap_path = os.path.join(RESULTS_DIR, "overfitting_analysis.png")
plt.savefig(gap_path, dpi=300, bbox_inches='tight')
print(f"✓ Analyse overfitting: {gap_path}")

# ===== STATISTIQUES =====
print("\n" + "="*70)
print("📊 STATISTIQUES D'ENTRAÎNEMENT")
print("="*70)

print(f"\n🏆 MEILLEUR MODÈLE:")
print(f"   Epoch: {best_epoch_acc}")
print(f"   Valid Accuracy: {best_valid_acc:.2f}%")
print(f"   Valid Loss: {valid_loss[best_epoch_acc-1]:.3f}")
print(f"   Train Accuracy: {train_acc[best_epoch_acc-1]:.2f}%")
print(f"   Écart Train/Valid: {train_acc[best_epoch_acc-1] - best_valid_acc:.1f}%")

print(f"\n📈 PROGRESSION:")
print(f"   Epoch 1 → Epoch {best_epoch_acc}:")
print(f"     Valid Accuracy: {valid_acc[0]:.1f}% → {best_valid_acc:.1f}% (+{best_valid_acc-valid_acc[0]:.1f}%)")
print(f"     Valid Loss: {valid_loss[0]:.2f} → {valid_loss[best_epoch_acc-1]:.2f}")

print(f"\n⚠️  SURAPPRENTISSAGE:")
max_gap = max(gap)
max_gap_epoch = epochs[gap.index(max_gap)]
print(f"   Écart maximum: {max_gap:.1f}% (Epoch {max_gap_epoch})")
if max_gap > 20:
    print(f"   🔴 Surapprentissage sévère détecté!")
elif max_gap > 10:
    print(f"   🟠 Surapprentissage modéré")
else:
    print(f"   🟢 Pas de surapprentissage")

print("\n" + "="*70)
print("✅ VISUALISATION TERMINÉE")
print("="*70)
print(f"\n📁 Graphiques sauvegardés dans: {RESULTS_DIR}/")
print(f"   - learning_curves.png")
print(f"   - overfitting_analysis.png")