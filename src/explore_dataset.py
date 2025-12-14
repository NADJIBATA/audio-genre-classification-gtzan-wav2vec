
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Chemins
DATA_PATH = Path("data/raw/Data/genres_original")
GENRES = ["blues", "classical", "country", "disco", "hiphop", 
          "jazz", "metal", "pop", "reggae", "rock"]

print("=" * 70)
print("🎵 EXPLORATION DU DATASET GTZAN")
print("=" * 70)




file_stats = []
corrupted_files = []

for genre in GENRES:
    genre_path = DATA_PATH / genre
    wav_files = list(genre_path.glob("*.wav"))
    
    print(f"Genre: {genre:12s} → {len(wav_files):3d} fichiers")
    
    for wav_file in wav_files:
        try:
            # Charger uniquement les métadonnées (rapide)
            duration = librosa.get_duration(path=str(wav_file))
            sr = librosa.get_samplerate(str(wav_file))
            file_size = wav_file.stat().st_size / (1024 * 1024)  # MB
            
            file_stats.append({
                'genre': genre,
                'filename': wav_file.name,
                'duration': duration,
                'sample_rate': sr,
                'file_size_mb': file_size
            })
        except Exception as e:
            print(f"  ⚠️  Fichier corrompu: {wav_file.name} - {str(e)}")
            corrupted_files.append({'genre': genre, 'filename': wav_file.name, 'error': str(e)})

# Créer DataFrame
df = pd.DataFrame(file_stats)

print(f"\n✅ Total de fichiers valides: {len(df)}/1000")
print(f"❌ Fichiers corrompus: {len(corrupted_files)}")

if corrupted_files:
    print("\n⚠️  Liste des fichiers corrompus:")
    for file in corrupted_files:
        print(f"   - {file['genre']}/{file['filename']}")



print(f"\nDurée moyenne: {df['duration'].mean():.2f}s")
print(f"Durée min: {df['duration'].min():.2f}s")
print(f"Durée max: {df['duration'].max():.2f}s")
print(f"Écart-type: {df['duration'].std():.2f}s")

# Fichiers avec durée anormale
abnormal_duration = df[(df['duration'] < 29) | (df['duration'] > 31)]
if len(abnormal_duration) > 0:
    print(f"\n⚠️  {len(abnormal_duration)} fichiers avec durée anormale:")
    print(abnormal_duration[['genre', 'filename', 'duration']])


print("\n" + "=" * 70)
print("🎚️  3. ANALYSE DES SAMPLE RATES")
print("=" * 70)

sample_rates = df['sample_rate'].value_counts()
print("\nDistribution des sample rates:")
print(sample_rates)

if len(sample_rates) > 1:
    print("\n⚠️  ATTENTION: Sample rates différents détectés!")
    print("   → Il faudra resampler à 16kHz pour WAV2VEC")
else:
    print(f"\n✅ Tous les fichiers ont le même sample rate: {sample_rates.index[0]} Hz")

print("\n" + "=" * 70)
print("🎭 4. DISTRIBUTION PAR GENRE")
print("=" * 70)

genre_counts = df['genre'].value_counts().sort_index()
print("\nNombre de fichiers par genre:")
print(genre_counts)

# Vérifier l'équilibre
is_balanced = all(genre_counts == 100)
if is_balanced:
    print("\n✅ Dataset parfaitement équilibré (100 fichiers par genre)")
else:
    print("\n⚠️  Dataset déséquilibré!")


print("\n" + "=" * 70)
print("📈 5. GÉNÉRATION DES VISUALISATIONS")
print("=" * 70)

# Créer dossier pour les figures
os.makedirs("results/figures", exist_ok=True)

# Figure 1: Distribution des durées
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histogramme global
axes[0, 0].hist(df['duration'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(30, color='red', linestyle='--', label='30s (attendu)')
axes[0, 0].set_xlabel('Durée (secondes)')
axes[0, 0].set_ylabel('Nombre de fichiers')
axes[0, 0].set_title('Distribution des durées - Global')
axes[0, 0].legend()

# Boxplot par genre
df.boxplot(column='duration', by='genre', ax=axes[0, 1])
axes[0, 1].set_xlabel('Genre')
axes[0, 1].set_ylabel('Durée (secondes)')
axes[0, 1].set_title('Distribution des durées par genre')
plt.sca(axes[0, 1])
plt.xticks(rotation=45)

# Distribution des tailles de fichiers
axes[1, 0].hist(df['file_size_mb'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1, 0].set_xlabel('Taille (MB)')
axes[1, 0].set_ylabel('Nombre de fichiers')
axes[1, 0].set_title('Distribution des tailles de fichiers')

# Barplot des genres
genre_counts.plot(kind='bar', ax=axes[1, 1], color='skyblue', edgecolor='black')
axes[1, 1].set_xlabel('Genre')
axes[1, 1].set_ylabel('Nombre de fichiers')
axes[1, 1].set_title('Nombre de fichiers par genre')
axes[1, 1].axhline(100, color='red', linestyle='--', label='Équilibre parfait')
axes[1, 1].legend()
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('results/figures/01_dataset_statistics.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: results/figures/01_dataset_statistics.png")
plt.show()



audio_features = []

for genre in GENRES:
    genre_path = DATA_PATH / genre
    wav_files = sorted(list(genre_path.glob("*.wav")))[:5]  # 5 premiers
    
    for wav_file in tqdm(wav_files, desc=f"Analyse {genre}", leave=False):
        try:
            # Charger l'audio
            y, sr = librosa.load(wav_file, sr=None, duration=30)
            
            # Extraire des features basiques
            features = {
                'genre': genre,
                'filename': wav_file.name,
                'rms_energy': float(np.sqrt(np.mean(y**2))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
                'tempo': float(librosa.feature.tempo(y=y, sr=sr)[0])
            }
            audio_features.append(features)
            
        except Exception as e:
            print(f"  Erreur sur {wav_file.name}: {e}")

df_features = pd.DataFrame(audio_features)

print("\n📊 Statistiques des features audio (échantillon):")
print(df_features.groupby('genre')[['rms_energy', 'tempo', 'spectral_centroid']].mean())


print("\n" + "=" * 70)
print("🖼️  7. GÉNÉRATION DES SPECTROGRAMMES")
print("=" * 70)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, genre in enumerate(GENRES):
    genre_path = DATA_PATH / genre
    example_file = list(genre_path.glob("*.wav"))[0]  # Premier fichier
    
    try:
        y, sr = librosa.load(example_file, sr=22050, duration=5)  # 5 premières secondes
        
        # Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', 
                                       ax=axes[idx], cmap='viridis')
        axes[idx].set_title(f'{genre.capitalize()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
        
    except Exception as e:
        axes[idx].text(0.5, 0.5, f'Erreur: {genre}', ha='center', va='center')
        axes[idx].set_title(genre)

plt.tight_layout()
plt.savefig('results/figures/02_spectrograms_by_genre.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: results/figures/02_spectrograms_by_genre.png")
plt.show()




# Sauvegarder les statistiques
df.to_csv('data/processed/file_metadata.csv', index=False)
print("✅ Sauvegardé: data/processed/file_metadata.csv")

if corrupted_files:
    pd.DataFrame(corrupted_files).to_csv('data/processed/corrupted_files.csv', index=False)
    print("✅ Sauvegardé: data/processed/corrupted_files.csv")

if len(df_features) > 0:
    df_features.to_csv('data/processed/audio_features_sample.csv', index=False)
    print("✅ Sauvegardé: data/processed/audio_features_sample.csv")

