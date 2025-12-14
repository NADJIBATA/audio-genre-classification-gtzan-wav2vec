"""
Baseline CNN pour classification de genres musicaux GTZAN
Architecture : Mel Spectrograms + CNN 2D avec Global Average Pooling
Version corrigée : 654K paramètres au lieu de 84M
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device utilisé: {DEVICE}")

# Hyperparamètres
CONFIG = {
    'sample_rate': 22050,
    'duration': 30,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,
    'seed': 42
}

# Reproductibilité
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['seed'])

print("=" * 70)
print("🎵 BASELINE CNN - GTZAN MUSIC GENRE CLASSIFICATION")
print("=" * 70)

# ============================================================================
# 1. DATASET CLASS
# ============================================================================

class GTZANDataset(Dataset):
    """Dataset PyTorch pour GTZAN avec Mel Spectrograms"""
    
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        # Encoder les labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['genre'])
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"   Dataset chargé: {len(self.df)} fichiers")
        print(f"   Classes: {self.label_encoder.classes_}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Charger le fichier audio
        audio_path = self.df.iloc[idx]['filepath'].replace('\\', '/')
        label = self.labels[idx]
        
        # Calculer le Mel spectrogram
        mel_spec = self._compute_mel_spectrogram(audio_path)
        
        # Convertir en tensor
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # [1, n_mels, time]
        label = torch.LongTensor([label])
        
        return mel_spec, label
    
    def _compute_mel_spectrogram(self, audio_path):
        """Calcule le Mel spectrogram d'un fichier audio"""
        # Charger l'audio
        y, sr = librosa.load(audio_path, sr=CONFIG['sample_rate'], 
                            duration=CONFIG['duration'])
        
        # Pad si nécessaire
        target_length = CONFIG['sample_rate'] * CONFIG['duration']
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Calculer Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr,
            n_mels=CONFIG['n_mels'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length']
        )
        
        # Convertir en dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normaliser
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        return mel_spec_norm

# ============================================================================
# 2. MODÈLE CNN AVEC GLOBAL AVERAGE POOLING
# ============================================================================

class MusicGenreCNN(nn.Module):
    """CNN avec Global Average Pooling - 654K paramètres"""
    
    def __init__(self, num_classes=10):
        super(MusicGenreCNN, self).__init__()
        
        print("   🏗️  Construction du modèle avec Global Average Pooling...")
        
        # Block 1: 1 → 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 2: 32 → 64 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 3: 64 → 128 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 4: 128 → 256 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Stats
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   ✅ Total paramètres: {total_params:,}")
        print(f"   ✅ Paramètres entraînables: {total_params:,}")
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================================
# 3. FONCTIONS D'ENTRAÎNEMENT
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne le modèle pour une epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.squeeze().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    """Évalue le modèle"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validation', leave=False):
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

# ============================================================================
# 4. SCRIPT PRINCIPAL
# ============================================================================

def main():
    print("\n🔧 Correction des chemins pour Linux/Colab...")
    for split in ['train', 'val', 'test']:
        csv_path = f'data/splits/{split}.csv'
        df = pd.read_csv(csv_path)
        df['filepath'] = df['filepath'].str.replace('\\', '/')
        df.to_csv(csv_path, index=False)
    print("   ✅ Chemins corrigés\n")
    
    print("\n📂 1. Chargement des datasets...")
    
    train_dataset = GTZANDataset('data/splits/train.csv')
    val_dataset = GTZANDataset('data/splits/val.csv')
    test_dataset = GTZANDataset('data/splits/test.csv')
    
    os.makedirs('models/baseline', exist_ok=True)
    with open('models/baseline/label_encoder.pkl', 'wb') as f:
        pickle.dump(train_dataset.label_encoder, f)
    print("   ✅ Label encoder sauvegardé")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=0)
    
    print("\n🏗️  2. Construction du modèle...")
    model = MusicGenreCNN(num_classes=train_dataset.num_classes).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total paramètres: {total_params:,}")
    print(f"   Paramètres entraînables: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    print("\n🏋️  3. Entraînement...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': CONFIG
            }, 'models/baseline/best_model.pth')
            print("   ✅ Meilleur modèle sauvegardé")
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⏹️  Early stopping à l'epoch {epoch+1}")
            break
    
    with open('models/baseline/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print("\n📊 4. Évaluation sur le test set...")
    
    checkpoint = torch.load('models/baseline/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)
    
    print(f"\n🎯 Résultats finaux:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    
    print("\n📋 Rapport de classification:")
    print(classification_report(test_labels, test_preds, 
                               target_names=train_dataset.label_encoder.classes_))
    
    # Matrice de confusion
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=train_dataset.label_encoder.classes_,
               yticklabels=train_dataset.label_encoder.classes_)
    plt.title('Matrice de Confusion - Baseline CNN')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/04_baseline_confusion_matrix.png', dpi=300)
    print("   ✅ Matrice de confusion sauvegardée")
    
    # Courbes d'apprentissage
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Courbes de Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title("Courbes d'Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/figures/05_baseline_training_curves.png', dpi=300)
    print("   ✅ Courbes d'apprentissage sauvegardées")
    
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'f1_score_macro': f1_score(test_labels, test_preds, average='macro'),
        'classification_report': classification_report(test_labels, test_preds, 
                                                       target_names=train_dataset.label_encoder.classes_,
                                                       output_dict=True)
    }
    
    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/baseline_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "=" * 70)
    print("✨ ENTRAÎNEMENT DU BASELINE TERMINÉ!")
    print("=" * 70)
    print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
    print(f"📊 F1-Score (macro): {results['f1_score_macro']:.4f}")

if __name__ == '__main__':
    main()