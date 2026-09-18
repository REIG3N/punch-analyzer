# Punch Analyzer

Analyse de la mécanique de frappe en boxe anglaise à partir d'une vidéo (sac ou shadowboxing, caméra fixe), pour évaluer objectivement la technique d'un pratiquant seul.

## Stack

- Python
- MediaPipe (Pose Landmarker)
- OpenCV
- scipy / scikit-learn

## État actuel

Extraction des landmarks de pose depuis une vidéo (export CSV) et détection des coups par pics de vitesse du poignet. Classification en cours de développement.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le modèle MediaPipe (`pose_landmarker_full.task`) n'est pas versionné dans ce repo — à télécharger depuis [la documentation officielle MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) et placer dans `data/`.

## Usage

```bash
# Extraction des landmarks vers CSV
python src/punch_analyzer/main.py

# Détection des coups (pics de vitesse du poignet) à partir du CSV
python -m punch_analyzer.strike_detection

# Outil de debug : rejoue la vidéo avec squelette, vitesse instantanée et
# coups détectés superposés (lecteur OpenCV natif, pas de frontend)
python -m punch_analyzer.debug_viewer --video video/videoplayback.mp4
```

## Licence

Projet personnel, portfolio en cours de développement.
