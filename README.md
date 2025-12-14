# Audio Genre Classification (GTZAN) avec Wav2Vec 2.0 (Fairseq)

Ce dépôt contient un pipeline de **classification de genres musicaux** sur **GTZAN (10 genres)**, avec :
- une **baseline CNN** (mél-spectrogrammes),
- Wav2Vec 2.0 en **fine-tuning complet (FFT)**,
- Wav2Vec 2.0 en **head-only fine-tuning (HFT)** (backbone figé).

> Les sorties (rapports, matrices de confusion, courbes d’apprentissage) sont déjà exportées dans `results/`.

---

## Structure du projet



.
├── src
│ ├── config.py
│ ├── create_splits.py
│ ├── dataset_wav2vec.py
│ ├── explore_dataset.py
│ ├── train_baseline.py
│ ├── train_wav2vec_FFT.py
│ ├── train_wav2vec_HFT.py
│ ├── utils.py
│ └── preprocessing/
│ ├── remove_corrupted_audio.py
│ ├── check_audio_file.py
│ ├── regenerate_labels.py
│ ├── fix_labels_from_tsv.py
│ ├── train_with_cleanup.py
│ ├── test_load_model.py
│ ├── test_model.py
│ ├── evaluate_wav2vec_testset.py
│ ├── generate_confusion_matrix.py
│ └── plot_learning_curves.py
├── data
│ └── processed/
│ ├── file_metadata.csv
│ ├── corrupted_files.csv
│ └── audio_features_sample.csv
└── results
├── FFT_results/
│ ├── classification_report.txt
│ ├── confusion_matrix.png
│ ├── learning_curves.png
│ ├── overfitting_analysis.png
│ └── errors_detail.txt
├── HFT_results/
│ ├── classification_report.txt
│ ├── Hconfusion_matrix.png
│ ├── Hlearning_curves.png
│ ├── Hoverfitting_analysis.png
│ └── errors_detail.txt
└── figures/
├── 01_dataset_statistics.png
├── 02_spectrograms_by_genre.png
└── 03_train_val_test_splits.png


---

## Prérequis

### Recommandé (important)
Fairseq peut être sensible aux versions récentes de Python. **Recommandation : Python 3.10** (ou 3.9) dans un environnement isolé.

### Dépendances (indicatives)
- Python 3.9/3.10
- PyTorch + torchaudio
- fairseq
- numpy, pandas, scikit-learn
- librosa, soundfile
- matplotlib (et éventuellement seaborn)

Exemple (à adapter selon ton GPU/OS) :

```bash
conda create -n gtzan-w2v python=3.10 -y
conda activate gtzan-w2v

# PyTorch (exemple)
pip install torch torchaudio

# Fairseq
pip install fairseq

# Outils data + plots
pip install numpy pandas scikit-learn librosa soundfile matplotlib seaborn

Données : GTZAN

Télécharger GTZAN (non inclus dans le dépôt).

Placer les fichiers audio dans le chemin attendu par les scripts :

data/raw/Data/genres_original/<genre>/*.wav


Genres attendus :
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

1) Nettoyage + métadonnées + splits
(Optionnel) Nettoyage / détection de fichiers corrompus

Selon ton workflow, tu peux utiliser les scripts dans src/preprocessing/.

Exemples :

python src/preprocessing/remove_corrupted_audio.py
python src/preprocessing/check_audio_file.py

Création des splits (train/val/test)

Le script create_splits.py s’appuie sur data/processed/file_metadata.csv et produit des splits sous data/splits/.

python src/create_splits.py

2) Préparation du format Fairseq (manifest TSV + labels)

Le script dataset_wav2vec.py prépare un dataset Fairseq dans data/fairseq/ :

copie des audios vers data/fairseq/audio/

création de train.tsv, valid.tsv, test.tsv

création de train.labels, valid.labels, test.labels

création de labels.json et dict.labels.txt

python src/dataset_wav2vec.py

3) Entraînement
Baseline CNN
python src/train_baseline.py

Wav2Vec 2.0 — Fine-tuning complet (FFT)
python src/train_wav2vec_FFT.py

Wav2Vec 2.0 — Head-only fine-tuning (HFT)
python src/train_wav2vec_HFT.py


Astuce : si tes scripts appellent fairseq-train, assure-toi d’avoir un checkpoint Wav2Vec2 compatible Fairseq (.pt) et ajuste les chemins/paramètres dans train_wav2vec_*.py.

4) Évaluation & visualisations
Tester / évaluer
python src/preprocessing/test_model.py
python src/preprocessing/evaluate_wav2vec_testset.py

Générer matrice de confusion et courbes
python src/preprocessing/generate_confusion_matrix.py
python src/preprocessing/plot_learning_curves.py


Les sorties sont (ou seront) sauvegardées dans results/FFT_results et results/HFT_results.

Résultats (déjà exportés)

FFT : accuracy ≈ 0.87 (voir results/FFT_results/classification_report.txt)

HFT : résultats disponibles dans results/HFT_results/classification_report.txt

Difficultés rencontrées (note)

Compatibilité Fairseq / versions récentes de Python : certains environnements récents peuvent casser l’installation ou l’exécution.

Solution pratique : utiliser conda + Python 3.10 (ou 3.9), et installer Fairseq dans cet environnement isolé.

En cas d’erreurs persistantes : pin des versions (torch/fairseq) et installation propre dans un env neuf.

Référence du code

Repo :

https://github.com/NADJIBATA/audio-genre-classification-gtzan-wav2vec


---

### Bonus (comme tu l’avais demandé pour ton rapport LaTeX)

À coller **après ta conclusion** pour afficher le lien GitHub + une section “difficultés” :

```latex
\subsection*{Code source}
Le code complet du projet est disponible sur GitHub :
\url{https://github.com/NADJIBATA/audio-genre-classification-gtzan-wav2vec}

\section{Difficultés rencontrées}
\subsection{Compatibilité Fairseq / versions récentes de Python}
Une difficulté importante a concerné l’implémentation avec \texttt{fairseq} :
selon l’environnement, certaines versions récentes de Python peuvent poser des
problèmes de compatibilité (installation, dépendances, exécution).
Pour stabiliser le pipeline, nous avons utilisé un environnement isolé (par ex.
\texttt{conda}) avec une version de Python plus adaptée (ex. Python~3.10) et des
versions cohérentes des bibliothèques (PyTorch/Fairseq), ce qui a permis de
reproduire l’entraînement et l’évaluation de manière fiable.
