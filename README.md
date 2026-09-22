# Punch Analyzer

Analyse de la mécanique de frappe en boxe anglaise à partir d'une vidéo (sac ou shadowboxing, caméra fixe), pour évaluer objectivement la technique d'un pratiquant seul.

## Stack

- Python
- MediaPipe (Pose Landmarker)
- OpenCV
- scipy / scikit-learn

## État actuel

Extraction des landmarks de pose depuis une vidéo (export CSV) et détection des coups jab/cross (Ticket 2). Un coup est un pic local d'`extension_ratio` du bras (distance épaule-poignet / longueur du bras, invariant à la distance caméra), confirmé par un filtre géométrique (hauteur, direction, vitesse) et un arbitrage croisé gauche/droite (un seul bras frappe à la fois). Classification par type de coup (crochet, uppercut) et calcul de metrics techniques : hors scope, réservés à un ticket futur.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installe le paquet `punch_analyzer` (layout `src/`) en mode éditable, pour que `python -m punch_analyzer.<module>` fonctionne depuis n'importe quel répertoire sans manipuler `PYTHONPATH`.

Le modèle MediaPipe (`pose_landmarker_full.task`) n'est pas versionné dans ce repo — à télécharger depuis [la documentation officielle MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) et placer dans `data/`.

## Usage

```bash
# Extraction des landmarks vers CSV. --start-ms/--end-ms coupent la vidéo ;
# --buffer-ms extrait une marge avant --start-ms (jamais comptée dans le score)
# pour que le lissage Savitzky-Golay ne mange pas le premier coup, pile au bord
# du signal extrait.
python -m punch_analyzer.landmark_extraction --video video/ma_video.mp4 --start-ms 3500 --buffer-ms 2500

# Détection des coups (pics d'extension_ratio) à partir du CSV
python -m punch_analyzer.strike_detection --video video/ma_video.mp4

# Outil de debug : rejoue la vidéo avec squelette, vitesse/extension/angle par
# main affichés en continu (pas seulement sur un coup détecté), et coups
# confirmés/rejetés superposés (lecteur OpenCV natif, pas de frontend)
python -m punch_analyzer.debug_viewer --video video/ma_video.mp4 --start-ms 3500 --buffer-ms 2500

# Score combo par combo sur une vidéo chorégraphiée à séquence connue (JSON
# ordonné {name, left, right} par combo, cf. data/combos_jab_cross.json) --
# voir "Limites connues" plus bas avant de lire le score agrégé
python -m punch_analyzer.combo_comparator --video video/ma_video.mp4 --combos data/combos_jab_cross.json --buffer-ms 2500

# Inspection brute du signal (candidats d'extension dans une plage de temps,
# courbe extension_ratio frame par frame, coups à extension quasi parfaite
# mais vitesse sous le seuil) pour diagnostiquer un coup raté avant de
# retoucher un seuil
python -m punch_analyzer.signal_inspector --help
```

## Limites connues

**Scope de la détection** : jab/cross uniquement (pic d'`extension_ratio` + filtre géométrique de hauteur/direction/vitesse + arbitrage croisé gauche/droite). Crochets et uppercuts ne sont pas détectés par construction — une trajectoire latérale (crochet) ou bas-vers-haut (uppercut) ne satisfait pas le filtre de direction actuel. Explicitement hors scope de ce ticket, réservé à une itération future (classification par type de coup).

**Segmentation en fenêtres d'activité (`combo_comparator`)** : le seuil de segmentation (`activity_threshold`/`min_window_ms`/`min_silence_ms`) n'a jamais été recalibré après sa valeur par défaut initiale. Sur la vidéo de test chorégraphiée (30 combos attendus), la segmentation automatique en détecte 23 — désalignement persistant, cause probable des micro-silences internes à un combo (jitter de tracking) trop courtes pour rester sous `min_silence_ms` mais suffisantes pour fragmenter/fusionner des fenêtres. **Le score agrégé produit par `combo_comparator` n'est donc pas fiable tel quel.** Utile en diagnostic combo par combo (avec vérification manuelle des bornes de chaque fenêtre via `signal_inspector`/`debug_viewer`), pas comme métrique de qualité globale du détecteur.

**Hypothèses testées et invalidées pendant ce ticket** (tracées ici pour ne pas les retester à l'identique) :
- *Mirroring gauche/droite* (vidéo enregistrée en mode selfie, labels MediaPipe inversés) — rejeté. L'arbitrage croisé favorise systématiquement la bonne main sur les données observées, à l'exception d'un cas isolé.
- *Démarrage à froid du filtre de lissage après un repos, comme cause principale des coups ratés* — rejeté. Un contre-exemple existe à 27.08s : un coup raté sans repos préalable, alors que l'hypothèse prédisait le contraire.
- *Asymétrie de vitesse gauche/droite justifiant un `min_peak_speed` séparé par main* — testé (distribution mesurée, séparation médiane/écart-type calculée), écart insuffisant pour être qualifié de net selon le critère retenu (médianes séparées d'au moins un écart-type). `min_peak_speed` reste un seuil global.

**Occlusion du bras arrière (monoculaire, position côté ouvert face caméra)** : limite structurelle d'une caméra unique, pas un bug. Le filtrage par `visibility` et le lissage Savitzky-Golay réduisent les faux positifs pendant l'occlusion mais ne garantissent pas la détection des vrais coups de ce bras si la caméra ne le voit pas au moment clé.

**Incompatibilité avec les vidéos à coupures de plan** (testé sur `video/videoplayback.mp4`, un montage/promo, pas une prise continue) : hors scope. L'architecture suppose une caméra fixe sans coupure ; MediaPipe en mode `VIDEO` track par continuité temporelle et casse sur un changement de plan (perte de tracking, faux raccords de position). Pas une régression de ce ticket — une hypothèse d'architecture que cette vidéo précise ne respecte pas.

## Licence

Projet personnel, portfolio en cours de développement.
