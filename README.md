🎵 Audio Genre Classification (GTZAN) avec Wav2Vec 2.0 & CNN

Ce dépôt propose un pipeline complet pour la classification de genres musicaux (10 classes du dataset GTZAN) via :

🎧 Une baseline CNN basée sur les mél-spectrogrammes

🔥 Wav2Vec 2.0 fine-tuning complet (FFT)

❄️ Wav2Vec 2.0 head-only fine-tuning (HFT) (backbone gelé)

Toutes les sorties (rapports, matrices de confusion, courbes d’apprentissage) sont déjà générées et disponibles dans results/.

📂 Structure du projet
.
├── src/
│   ├── config.py
│   ├── create_splits.py
│   ├── dataset_wav2vec.py
│   ├── explore_dataset.py
│   ├── train_baseline.py
│   ├── train_wav2vec_FFT.py
│   ├── train_wav2vec_HFT.py
│   ├── utils.py
│   └── preprocessing/
│       ├── remove_corrupted_audio.py
│       ├── check_audio_file.py
│       ├── regenerate_labels.py
│       ├── fix_labels_from_tsv.py
│       ├── train_with_cleanup.py
│       ├── test_load_model.py
│       ├── test_model.py
│       ├── evaluate_wav2vec_testset.py
│       ├── generate_confusion_matrix.py
│       └── plot_learning_curves.py
│
├── data/
│   └── processed/
│       ├── file_metadata.csv
│       ├── corrupted_files.csv
│       └── audio_features_sample.csv
│
└── results/
    ├── FFT_results/
    │   ├── classification_report.txt
    │   ├── confusion_matrix.png
    │   ├── learning_curves.png
    │   ├── overfitting_analysis.png
    │   └── errors_detail.txt
    │
    ├── HFT_results/
    │   ├── classification_report.txt
    │   ├── Hconfusion_matrix.png
    │   ├── Hlearning_curves.png
    │   ├── Hoverfitting_analysis.png
    │   └── errors_detail.txt
    │
    └── figures/
        ├── 01_dataset_statistics.png
        ├── 02_spectrograms_by_genre.png
        └── 03_train_val_test_splits.png

⚙️ Prérequis
✔️ Recommandé

Fairseq ne supporte pas encore officiellement les versions récentes de Python.

👉 Utiliser Python 3.9 ou 3.10 dans un environnement isolé (conda ou venv).

✔️ Dépendances principales

torch, torchaudio

fairseq

numpy, pandas, scikit-learn

librosa, soundfile

matplotlib, seaborn

Installation type
conda create -n gtzan-w2v python=3.10 -y
conda activate gtzan-w2v

pip install torch torchaudio
pip install fairseq

pip install numpy pandas scikit-learn librosa soundfile matplotlib seaborn

🎼 Données : GTZAN

Télécharger GTZAN (Kaggle).

L’arborescence attendue :

data/raw/Data/genres_original/<genre>/*.wav


Genres :

blues, classical, country, disco, hiphop,
jazz, metal, pop, reggae, rock

🛠️ 1) Préprocessing & Splits
Nettoyage (optionnel)
python src/preprocessing/remove_corrupted_audio.py
python src/preprocessing/check_audio_file.py

Création des splits (train/val/test)
python src/create_splits.py

🗂️ 2) Préparation du dataset Fairseq

Génération des manifests TSV + labels + copie des fichiers dans data/fairseq/.

python src/dataset_wav2vec.py


Cela produit :

data/fairseq/
├── audio/
├── train.tsv      train.labels
├── valid.tsv      valid.labels
├── test.tsv       test.labels
├── labels.json
└── dict.labels.txt

🧠 3) Entraînement
🔵 Baseline CNN
python src/train_baseline.py


Sorties dans :

results/baseline/
models/baseline/

🔥 Wav2Vec 2.0 — Fine-Tuning Complet (FFT)
python src/train_wav2vec_FFT.py

❄️ Wav2Vec 2.0 — Head-Only Fine-Tuning (HFT)

(backbone gelé)

python src/train_wav2vec_HFT.py

📊 4) Évaluation & Visualisation
Tests / prédictions
python src/preprocessing/test_model.py
python src/preprocessing/evaluate_wav2vec_testset.py

Matrice de confusion
python src/preprocessing/generate_confusion_matrix.py

Courbes d’apprentissage
python src/preprocessing/plot_learning_curves.py

🏁 Résultats (déjà exportés)
⭐ Fine-Tuning Complet (FFT)

✔ Accuracy ≈ 0.87
✔ Meilleur modèle

❄ Head-Only (HFT)

✔ Accuracy ≈ 0.75

🔵 Baseline CNN

✔ Accuracy ≈ 0.73

Détails disponibles dans :

results/FFT_results/
results/HFT_results/

⚠️ Difficultés rencontrées
Compatibilité Fairseq / Python récent

Certaines versions (Python 3.11/3.12) cassent l'installation de Fairseq.

Solution :
✔ Utiliser Python 3.10
✔ Installer Fairseq dans un environnement isolé
✔ Vérifier la version de PyTorch compatible
