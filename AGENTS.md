# Punch Analyzer — Contexte projet

Document de contexte du projet pour les agents (AGENTS.md), maintenu en parallèle de `opencode.json`, `.zed/settings.json` et `.envrc`. Relu et complété à chaque étape significative.

---

## 1. Vue d'ensemble

Application de computer vision analysant la mécanique de frappe en boxe anglaise, à partir de **vidéo pré-enregistrée** (pas de temps réel). Projet portfolio destiné au recrutement technique — priorité à la démonstration de compétences full-stack. Auteur : pratiquant de boxe (expérience personnelle).

- **Cible** : pratiquant solo (sac de frappe ou shadowboxing) — évite l'identification d'athlète et le bruit d'occlusion tierce.
- **Objectif** : analyser le séquençage biomécanique (jambes → hanches → épaules → bras), donner un feedback de mécanique, afficher des stats de séance. Priorité bas du corps + tronc (génération de puissance), pas le tracking fin bras/main.
- **Phase 1** : tous les types de poings (jab, cross, crochet, uppercut). Pieds/genoux/coudes = Phase 2, différé.
- **Format cible** : PWA avec backend Python personnel, mono-utilisateur.

---

## 2. Architecture — décisions actées

Pipeline en 5 phases en série :
1. Extraction & stockage (MediaPipe → CSV)
2. Détection des coups (traitement du signal, pas ML)
3. Classification du type de coup (ML supervisé — Random Forest)
4. Scoring biomécanique (système de règles, pas ML)
5. Stats & progression (calcul, pas ML)

**Décisions structurantes :**
- Extension/impact comme ancrage temporel pour l'analyse rétrospective du séquençage.
- 2 caméras (Pose2Sim) différé — pertinent seulement si le monoculaire s'avère insuffisant ; **non concurrent** de MediaPipe (Pose2Sim consomme des keypoints 2D, ne détecte pas lui-même).
- 3 caméras écarté (hors de portée pour un pratiquant solo).
- Capteurs IMU (projet « Shadowbox IoT ») différé, scopé plus tard (horizon Master 2027-2029), pas abandonné.
- Backend from-scratch sur serveur/URL personnel, pas de service cloud tiers.
- Stockage vidéo (compressé + overlay landmarks) hébergé serveur plutôt qu'en local navigateur (évite les limites IndexedDB/OPFS d'une PWA).
- Infrastructure (backend, upload, dashboard) volontairement **mise de côté** — focus actuel = pipeline technique (extraction → détection → classification → scoring).

**Séparation ML vs règles (point clé) :**
- **Détection** (Phase 2) = traitement du signal, pas de ML.
- **Classification** (Phase 3, quel coup) = ML supervisé → Random Forest + apprentissage actif (faut de dataset labellisé public).
- **Scoring** (Phase 4, qualité) = système de règles codées depuis la littérature, PAS du ML (aucun dataset de scores humains).
- **Nuance** : le scoring intervient APRÈS la classification, jamais avant — les seuils dépendent du type de coup.

---

## 3. Environnement technique

- **IDE** : Zed avec basedpyright et Ruff (config `.zed/settings.json` au niveau projet).
- **OS** : Arch Linux, shell **bash**.
- **venv** : `direnv` configuré — `.envrc` = `source .venv/bin/activate`, activation automatique à l'entrée.
- **Structure** : `~/Projects/ppanalyz` avec `src/punch_analyzer/`, `tests/`, `data/`, `video/`, `.venv/`.
  - `src/punch_analyzer/main.py` = pipeline (exécutable) ; `__init__.py` = marqueur de package vide (aucun effet de bord à l'import).
- **Dépendances** (`requirements.txt`) : mediapipe, opencv-python, opencv-contrib-python, numpy, matplotlib, etc. — installées dans le `.venv`.
- **Git** : initialisé, branche `main`, pas de remote GitHub (aucun commit actuellement).
- **Modèle MediaPipe** : `data/pose_landmarker_full.task`, mode VIDEO.

---

## 4. Arbre technique (5 phases)

Légende : ✅ fait · 🟡 en cours · ⬜ à faire · ⚠️ décision ou limite bloquante

### Phase 1 — Extraction & stockage
- **1.1 MediaPipe** ✅ — `PoseLandmarker` mode VIDEO, `TRACKED_INDICES` (20 landmarks, visage/pouce exclus).
- **1.2 Lissage EMA** ✅ — `SMOOTHING_ALPHA = 0.7` sur x/y de chaque landmark, **réservé à l'affichage overlay** (le stockage reçoit les coordonnées brutes).
- **1.3 Stockage CSV** 🟡 (Ticket 1) — **décisions actées** : stocker les coordonnées **brutes** (1.3.1) ; **sauter** les frames sans détection, chaque ligne garde son `timestamp_ms` (1.3.2). Colonnes `frame_idx, timestamp_ms, landmark_id, x, y, z, visibility`. Écriture `csv.DictWriter` (`writeheader()` + `writerows()`). Validation 1.3.5 = lignes × frames_détectées × 20, pas de NaN, `timestamp_ms` strictement croissant. ⚠️ À faire : offset **temps absolu** du timestamp (repart actuellement de 0 malgré le seek à 120 s). Type des coordonnées : `NormalizedLandmark` = dataclass avec champs float Python natifs (via ctypes c_float) → `float()` de of exact, aucun risque de type non-natif.

### Phase 2 — Détection des coups
- **2.1** Vitesse doigt (`Δposition/Δt` via `numpy.diff`), lissage éventuel `savgol_filter` à évaluer empiriquement (pas par défaut).
- **2.2** `find_peaks` ancré sur **extension max/impact** ; réglage `height`/`distance` (durée d'un coup ~300-405 ms) ; validation manuelle faux+ / faux−.
- **2.3** Fenêtre de séquençage (cheville → genou → hanche → épaule → coude) ; largeur ~500 ms à ajuster ; sous-séries par articulation via filtrage `pandas` sur `frame_idx`.

### Phase 3 — Classification (quel coup ?)
- **3.1** Features cinématiques minimales : vitesse max doigt, angle du coude au pic (`arctan2`), durée du segment → `pandas.DataFrame` (X).
- **3.2** Labellisation (volume 20-100 coups à fixer, vidéos personnelles solo) : jab, cross, crochet av/arr, uppercut av/arr, « bruit ». Le label ne porte QUE sur le type, jamais sur la qualité.
- **3.3** Random Forest (`train_test_split` → `RandomForestClassifier` → `classification_report` par type de classe).
- **3.4** Apprentissage actif `UncertaintySampling` (врав Scikit-learn) — limité à l'incertitude, vérifier manuellement les fausses confiances.

### Phase 4 — Scoring biomécanique (règles, pas ML)
- **4.1** Séquençage proximal-distal : ordre des pics de vitesse angulaire (chville < genou < hanche < épaule < coude), tolérance au bruit.
- **4.2** Détection arm-punching : proxy angulaire (rotation épaule/rotation totale), approximation documentée du concept scientifique.
- **4.3** ⚠️ **DÉCISION BLOQUANTE** Transfert de poids / GRF non mesurable par MediaPipe seul (plateformes de force requises). Options : (a) proxy géométrique via déplacement horizontal des hanches, (b) report à l'intégration IMU.
- **4.4** Seuils par type de coup (table `dict[str, dict]`), tolérance ±10° minimum.

### Phase 5 — Stats & progression
- **5.1** `current_perf` : historique par type de coup, écart % à la moyenne (seuil minimal de séances à définir).
- **5.2** Corrélations inter-coups : chaînage si écart temporel < seuil « combinaison » (à dériver empiriquement).
- **5.3** Agrégation par séance : total, répartition, score moyen.

---

## 5. Base scientifique — règles biomécaniques sourcées

Recherche PubMed effectuée, comparée au modèle éditorial « The Punch Doctor » (créateur de contenu, non scientifique).

**Confirmé (fiable) :**
- Séquencage proximal-distal jambes→hanches→torse→buste→bras : Cheraghi et al. 2014 (extension cheville ~45%, genou ~60%, coude ~80%) ; Filimonov et al. 1985 (jambes : 38,6% experts vs 16,5% novices).
- Arm-punching comme margeur de niveau novice : Dinu & Louis 2020 (épaule cross 15,6% élites vs 29,1% juniors).
- Cycle étirement-raccourcissement (SSC) entraînable : Sánchez-Ramírez et al. 2025.
- Angle du genou optimal ~125-130° (+24,4% de puissance) — source non-PubMed (Power Cube, n=1, indicatif).

**Nuancé :**
- Rotation « simultanée » hanche-épaule pas forcément un défaut : Fuchs et al. 2018 (conservation de l'ordre proximal-distal des pics, même avec initiation simultanée). → la métrique doit porter sur l'ordre des pics.
- Modèle en 3 phases valide pour droits, 4 pour crochets (Lenetsky et al. 2020) — proche du découpage pédagogique mais non identique.

**Non mesurable avec MediaPipe :** transfert de poids / GRF (voir 4.3).

**Limites MediaPipe :** erreur d'angle 3-9° selon articulation ; caméra optimale à 45° (MAE ~13°) ; occlusion et rotations rapides réduisent la fidélité.

---

## 6. Méthode de travail (protocole utilisateur)

- **Protocole OTW (Objective → Ticket → Where-to-look)** ; tickets typés avec mots-clés à chercher, référence de doc, critère de validation comportementale — pas de pseudo-code fourni.
- Halving du temps en cas de blocage (plafonné 2-3 subdivions avant de basculer sur un exemple travaillé).
- Redivision si absence totale de piste (même avant le délai) ; surveiller que ça ne devienne pas un moyen de procrastiner.
- Constat : l'OTW fonctionne bien sur du tooling mais bute sur les fondamentaux de syntaxe/concept Python purs (besoin d'exemples travaillés plus tôt là-dessus).

---

## 7. Statut actuel

- **DONNÉES** : Phase 1.1 ✅, 1.2 ✅ (EMA) — pipeline monolithique refactorisée dans `main.py` (fichiers de travail).

- **Prochain ticket actionnable** : **Ticket 1 (Phase 1.3 — stockage CSV)**, deux décisions déjà actées (brut + sauter frames sans détection).
- Phases 2-5 cartographiées mais non commencées.

**Décisions restantes :**
- Volume/source du premier lot de labellisation (Phase 3.2.3).
- Format de stockage de l'historique de séances (Phase 5.1.1) — hors scope.
- 4.3 transfert de poids : proxy géométrique vs report IMU (bloquant sur la compat).

---

## Commandes

- Activation env : automatique (direnv) ; sinon `source .venv/bin/activate`.
- Lancement pipeline : `python src/punch_analyzer/main.py`.
- Lint : `ruff check .` (Ruff aussi formateur dans Zed, format on save).
- Tests : `pytest` (à installer ; `tests/` vide actuellement).