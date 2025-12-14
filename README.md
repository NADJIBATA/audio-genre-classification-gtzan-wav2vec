from pathlib import Path

readme = r"""# Audio Genre Classification (GTZAN) avec Wav2Vec 2.0 (Fairseq)

Ce projet implémente une **classification de genres musicaux** sur le dataset **GTZAN (10 genres)** en comparant :
- une **baseline CNN** sur **mél-spectrogrammes** ;
- **Wav2Vec 2.0** avec **fine-tuning complet** (backbone + tête de classification) ;
- **Wav2Vec 2.0** avec **tête seule** (backbone figé, seule la tête est entraînée).

## Résultats (test)

| Modèle | Représentation | Accuracy (test) | F1 macro |
|---|---|---:|---:|
| CNN (baseline) | Mél-spectrogrammes | 73.33% | 0.70 |
| Wav2Vec 2.0 – fine-tuning complet | Audio brut | 87% | 0.84 |
| Wav2Vec 2.0 – tête seule | Audio brut | 75% | 0.72 |

> Remarque : les valeurs proviennent du rapport/exports du projet (jeu de test de 100 extraits).

---

## Arborescence (haut niveau)

- `src/` : scripts d’entraînement, préparation dataset, utilitaires, génération de figures.
- `data/` : données (brutes et/ou pré-traitées), manifests/labels (selon votre exécution).
- `results/` : exports (rapports de classification, matrices de confusion, courbes d’apprentissage, etc.).

---

## Prérequis

### Recommandation (important)
Le framework **Fairseq** peut être sensible aux versions récentes de Python / dépendances.
Pour éviter les soucis de compatibilité :
- utilisez un environnement isolé (**conda** ou **venv**) ;
- privilégiez **Python 3.9 ou 3.10**.

### Dépendances (indicatif)
- `python`
- `torch`, `torchaudio`
- `fairseq`
- `numpy`, `pandas`, `scikit-learn`
- `librosa`, `soundfile`
- `matplotlib`

Exemple (à adapter selon votre OS/GPU) :

```bash
conda create -n gtzan-w2v python=3.10 -y
conda activate gtzan-w2v

pip install torch torchaudio
pip install fairseq
pip install numpy pandas scikit-learn librosa soundfile matplotlib
