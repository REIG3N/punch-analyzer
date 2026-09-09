# Punch Analyzer

Analyse de la mécanique de frappe en boxe anglaise à partir d'une vidéo (sac ou shadowboxing, caméra fixe), pour évaluer objectivement la technique d'un pratiquant seul.

## Stack

- Python
- MediaPipe (Pose Landmarker)
- OpenCV
- scipy / scikit-learn

## État actuel

Extraction des landmarks de pose depuis une vidéo, export en CSV. Détection et classification des coups en cours de développement.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le modèle MediaPipe (`pose_landmarker_full.task`) n'est pas versionné dans ce repo — à télécharger depuis [la documentation officielle MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) et placer dans `data/`.

## Usage

```bash
python src/punch_analyzer/main.py
```

## Licence

Projet personnel, portfolio en cours de développement.
