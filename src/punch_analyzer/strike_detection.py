import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from punch_analyzer.paths import CSV_OUTPUT_DIR, VIDEO_PATH, csv_path_for_video
from punch_analyzer.smoothing import smooth_series as savgol_smooth_series

LEFT_WRIST_ID = 15
RIGHT_WRIST_ID = 16
WRIST_LANDMARKS = {"left": LEFT_WRIST_ID, "right": RIGHT_WRIST_ID}

LEFT_SHOULDER_ID = 11
RIGHT_SHOULDER_ID = 12
LEFT_ELBOW_ID = 13
RIGHT_ELBOW_ID = 14
LEFT_HIP_ID = 23
RIGHT_HIP_ID = 24
ARM_LANDMARKS = {
    "left": {"shoulder": LEFT_SHOULDER_ID, "elbow": LEFT_ELBOW_ID, "wrist": LEFT_WRIST_ID},
    "right": {"shoulder": RIGHT_SHOULDER_ID, "elbow": RIGHT_ELBOW_ID, "wrist": RIGHT_WRIST_ID},
}

DEFAULT_MAX_GAP_FRAMES = 5
DEFAULT_MIN_STRIKE_INTERVAL_MS = 250.0

DEFAULT_MIN_VISIBILITY = 0.5
# Lissage Savitzky-Golay (non causal : la vidéo est déjà enregistrée en entier,
# pas de contrainte de temps réel) appliqué aux positions x/y avant de calculer
# vitesse et extension_ratio. window doit être impair.
DEFAULT_SAVGOL_WINDOW = 9
DEFAULT_SAVGOL_POLYORDER = 3

DEFAULT_EXTENSION_RATIO_THRESHOLD = 0.85
DEFAULT_EXTENSION_PEAK_PROMINENCE = 0.15
DEFAULT_HEAD_HEIGHT_TORSO_FRACTION = 1 / 3
DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG = 35.0
DEFAULT_MIN_PEAK_SPEED = 1.5
DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES = 5
DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES = DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES

DEFAULT_ACTIVITY_THRESHOLD = 0.3
DEFAULT_MIN_WINDOW_MS = 300.0
DEFAULT_MIN_SILENCE_MS = 1200.0


@dataclass
class Strike:
    hand: str
    frame_idx: int
    timestamp_ms: float
    speed: float
    geometric_pass: bool = False
    extension_ratio: float | None = None


def _fill_short_gaps(series: pd.Series, max_gap_frames: int) -> pd.Series:
    is_na = series.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_length = is_na.groupby(run_id).transform("sum")
    fillable = is_na & (run_length <= max_gap_frames)
    interpolated = series.interpolate(method="linear", limit_area="inside")
    return series.where(~fillable, interpolated)


def _mask_low_visibility(
    x: pd.Series, y: pd.Series, visibility: pd.Series, min_visibility: float
) -> tuple[pd.Series, pd.Series]:
    low_confidence = visibility < min_visibility
    return x.mask(low_confidence), y.mask(low_confidence)


def pivot_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Format large : une ligne par frame_idx, colonnes x_<id>/y_<id>/visibility_<id>."""
    wide = df.pivot(index="frame_idx", columns="landmark_id", values=["x", "y", "visibility"])
    wide.columns = [f"{field}_{int(landmark_id)}" for field, landmark_id in wide.columns]
    wide = wide.sort_index()
    timestamps = df.groupby("frame_idx")["timestamp_ms"].first().sort_index()
    wide.insert(0, "timestamp_ms", timestamps)
    return wide.reset_index()


def _landmark_series(wide: pd.DataFrame, field: str, landmark_id: int) -> pd.Series:
    column = f"{field}_{landmark_id}"
    if column in wide.columns:
        return wide[column]
    return pd.Series(np.nan, index=wide.index)


def _landmark_xy(
    wide: pd.DataFrame, landmark_id: int, min_visibility: float
) -> tuple[pd.Series, pd.Series]:
    x = _landmark_series(wide, "x", landmark_id)
    y = _landmark_series(wide, "y", landmark_id)
    visibility = _landmark_series(wide, "visibility", landmark_id)
    return _mask_low_visibility(x, y, visibility, min_visibility)


def _smoothed_xy(
    x: pd.Series,
    y: pd.Series,
    max_gap_frames: int,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[pd.Series, pd.Series]:
    x = _fill_short_gaps(x, max_gap_frames)
    y = _fill_short_gaps(y, max_gap_frames)
    x = savgol_smooth_series(x, savgol_window, savgol_polyorder)
    y = savgol_smooth_series(y, savgol_window, savgol_polyorder)
    return x, y


def _velocity(
    x: pd.Series,
    y: pd.Series,
    timestamp_ms: pd.Series,
    max_gap_frames: int,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[pd.Series, pd.Series]:
    x, y = _smoothed_xy(x, y, max_gap_frames, savgol_window, savgol_polyorder)

    dt_seconds = timestamp_ms.diff() / 1000.0
    vx = x.diff() / dt_seconds
    vy = y.diff() / dt_seconds
    return vx.where(dt_seconds > 0), vy.where(dt_seconds > 0)


def compute_torso_center_velocity(
    wide: pd.DataFrame,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    savgol_window: int = DEFAULT_SAVGOL_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> pd.DataFrame:
    left_x, left_y = _landmark_xy(wide, LEFT_SHOULDER_ID, min_visibility)
    right_x, right_y = _landmark_xy(wide, RIGHT_SHOULDER_ID, min_visibility)
    center_x = pd.concat([left_x, right_x], axis=1).mean(axis=1, skipna=True)
    center_y = pd.concat([left_y, right_y], axis=1).mean(axis=1, skipna=True)

    vx, vy = _velocity(
        center_x, center_y, wide["timestamp_ms"], max_gap_frames, savgol_window, savgol_polyorder
    )
    return pd.DataFrame({"frame_idx": wide["frame_idx"], "vx": vx, "vy": vy})


def compute_wrist_speed(
    df: pd.DataFrame,
    landmark_id: int,
    torso_velocity: pd.DataFrame | None = None,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    savgol_window: int = DEFAULT_SAVGOL_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> pd.DataFrame:
    """Vitesse (unités normalisées/s) d'un poignet, relative au centre-épaules si torso_velocity est fourni."""
    wrist = (
        df[df["landmark_id"] == landmark_id]
        .sort_values("frame_idx")
        .reset_index(drop=True)
        .copy()
    )
    wrist["x"], wrist["y"] = _mask_low_visibility(
        wrist["x"], wrist["y"], wrist["visibility"], min_visibility
    )
    vx, vy = _velocity(
        wrist["x"], wrist["y"], wrist["timestamp_ms"], max_gap_frames,
        savgol_window, savgol_polyorder,
    )

    if torso_velocity is not None:
        torso = torso_velocity.set_index("frame_idx")
        torso_vx = wrist["frame_idx"].map(torso["vx"]).fillna(0.0)
        torso_vy = wrist["frame_idx"].map(torso["vy"]).fillna(0.0)
        vx = vx - torso_vx
        vy = vy - torso_vy

    wrist["speed"] = np.sqrt(vx**2 + vy**2)
    return wrist


def compute_arm_geometry(
    wide: pd.DataFrame,
    hand: str,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    savgol_window: int = DEFAULT_SAVGOL_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> pd.DataFrame:
    """extension_ratio, hauteur du poignet et angle épaule->poignet par frame, pour le
    bras `hand`. Positions lissées (Savitzky-Golay) avant calcul, comme pour la
    vitesse : le jitter de tracking sur une frame isolée peut sinon suffire à faire
    dépasser extension_threshold à un bras qui ne frappe pas."""
    ids = ARM_LANDMARKS[hand]

    def smoothed(landmark_id: int) -> tuple[pd.Series, pd.Series]:
        x, y = _landmark_xy(wide, landmark_id, min_visibility)
        return _smoothed_xy(x, y, max_gap_frames, savgol_window, savgol_polyorder)

    shoulder_x, shoulder_y = smoothed(ids["shoulder"])
    elbow_x, elbow_y = smoothed(ids["elbow"])
    wrist_x, wrist_y = smoothed(ids["wrist"])

    other_shoulder_id = RIGHT_SHOULDER_ID if hand == "left" else LEFT_SHOULDER_ID
    other_shoulder_x, other_shoulder_y = smoothed(other_shoulder_id)
    hip_left_x, hip_left_y = smoothed(LEFT_HIP_ID)
    hip_right_x, hip_right_y = smoothed(RIGHT_HIP_ID)

    def dist(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    reach = dist(shoulder_x, shoulder_y, elbow_x, elbow_y) + dist(
        elbow_x, elbow_y, wrist_x, wrist_y
    )
    extension_ratio = dist(shoulder_x, shoulder_y, wrist_x, wrist_y) / reach

    shoulder_center_x = pd.concat([shoulder_x, other_shoulder_x], axis=1).mean(
        axis=1, skipna=True
    )
    shoulder_center_y = pd.concat([shoulder_y, other_shoulder_y], axis=1).mean(
        axis=1, skipna=True
    )
    hip_center_x = pd.concat([hip_left_x, hip_right_x], axis=1).mean(axis=1, skipna=True)
    hip_center_y = pd.concat([hip_left_y, hip_right_y], axis=1).mean(axis=1, skipna=True)
    torso_length = dist(shoulder_center_x, shoulder_center_y, hip_center_x, hip_center_y)

    angle_deg = np.degrees(
        np.arctan2((wrist_y - shoulder_y).abs(), (wrist_x - shoulder_x).abs())
    )

    return pd.DataFrame(
        {
            "frame_idx": wide["frame_idx"],
            "timestamp_ms": wide["timestamp_ms"],
            "extension_ratio": extension_ratio,
            "wrist_y": wrist_y,
            "shoulder_y": shoulder_y,
            "torso_length": torso_length,
            "angle_deg": angle_deg,
        }
    )


def _passes_geometric_thresholds(
    extension_ratio: float,
    wrist_y: float,
    shoulder_y: float,
    torso_length: float,
    angle_deg: float,
    peak_speed: float,
    extension_threshold: float,
    height_fraction: float,
    angle_threshold_deg: float,
    min_peak_speed: float,
) -> bool:
    if extension_ratio < extension_threshold:
        return False
    if wrist_y > shoulder_y + torso_length * height_fraction:
        return False
    if angle_deg > angle_threshold_deg:
        return False
    return peak_speed >= min_peak_speed


def detect_strikes_for_hand(
    geometry: pd.DataFrame,
    wrist_speed: pd.DataFrame,
    hand: str,
    min_interval_ms: float = DEFAULT_MIN_STRIKE_INTERVAL_MS,
    extension_prominence: float = DEFAULT_EXTENSION_PEAK_PROMINENCE,
    extension_threshold: float = DEFAULT_EXTENSION_RATIO_THRESHOLD,
    height_fraction: float = DEFAULT_HEAD_HEIGHT_TORSO_FRACTION,
    angle_threshold_deg: float = DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG,
    min_peak_speed: float = DEFAULT_MIN_PEAK_SPEED,
    confirm_window_frames: int = DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES,
) -> list[Strike]:
    """Détection sur les pics d'extension_ratio : un cycle aller-retour de coup ne
    produit qu'un seul maximum local d'extension (contrairement à la vitesse, qui en
    a deux, aller et retour, et double-compte). La vitesse sert de filtre secondaire
    (mesurée dans une fenêtre autour du pic d'extension), pas d'axe de détection."""
    valid = geometry.dropna(subset=["extension_ratio"])
    if valid.empty:
        return []

    median_dt_ms = valid["timestamp_ms"].diff().median()
    if not median_dt_ms or median_dt_ms <= 0:
        return []
    min_distance_frames = max(1, round(min_interval_ms / median_dt_ms))

    peak_positions, _ = find_peaks(
        valid["extension_ratio"].to_numpy(),
        distance=min_distance_frames,
        prominence=extension_prominence,
    )
    peaks = valid.iloc[peak_positions]

    speed_by_frame = wrist_speed.set_index("frame_idx")["speed"]

    strikes = []
    for row in peaks.itertuples():
        frame_idx = int(row.frame_idx)
        window = speed_by_frame[
            (speed_by_frame.index >= frame_idx - confirm_window_frames)
            & (speed_by_frame.index <= frame_idx + confirm_window_frames)
        ].dropna()
        peak_speed = float(window.max()) if not window.empty else 0.0

        geometric_pass = _passes_geometric_thresholds(
            row.extension_ratio, row.wrist_y, row.shoulder_y, row.torso_length,
            row.angle_deg, peak_speed, extension_threshold, height_fraction,
            angle_threshold_deg, min_peak_speed,
        )

        strikes.append(
            Strike(
                hand=hand,
                frame_idx=frame_idx,
                timestamp_ms=float(row.timestamp_ms),
                speed=peak_speed,
                geometric_pass=geometric_pass,
                extension_ratio=float(row.extension_ratio),
            )
        )
    return strikes


@dataclass
class ActivityWindow:
    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float


def segment_activity_windows(
    df: pd.DataFrame,
    activity_threshold: float = DEFAULT_ACTIVITY_THRESHOLD,
    min_window_ms: float = DEFAULT_MIN_WINDOW_MS,
    min_silence_ms: float = DEFAULT_MIN_SILENCE_MS,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    savgol_window: int = DEFAULT_SAVGOL_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
) -> list[ActivityWindow]:
    """Segmente la vidéo en fenêtres d'activité à partir de la vitesse (relative au
    tronc) du poignet le plus rapide par frame. Les silences plus courts que
    min_silence_ms sont pontés (hystérésis) pour ne pas fragmenter un combo qui
    ralentit brièvement en son milieu ; les silences plus longs (pauses entre
    combos) séparent bien deux fenêtres distinctes."""
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(
        wide, max_gap_frames, min_visibility, savgol_window, savgol_polyorder
    )
    left_speed = compute_wrist_speed(
        df, LEFT_WRIST_ID, torso_velocity, max_gap_frames, min_visibility,
        savgol_window, savgol_polyorder,
    ).set_index("frame_idx")["speed"]
    right_speed = compute_wrist_speed(
        df, RIGHT_WRIST_ID, torso_velocity, max_gap_frames, min_visibility,
        savgol_window, savgol_polyorder,
    ).set_index("frame_idx")["speed"]

    combined = pd.concat([left_speed, right_speed], axis=1).max(axis=1, skipna=True)
    combined = combined.fillna(0.0).sort_index()
    timestamps = wide.set_index("frame_idx")["timestamp_ms"].reindex(combined.index)

    median_dt_ms = timestamps.diff().median()
    if not median_dt_ms or median_dt_ms <= 0:
        return []
    min_silence_frames = max(1, round(min_silence_ms / median_dt_ms))
    min_window_frames = max(1, round(min_window_ms / median_dt_ms))

    active = combined > activity_threshold
    inactive = ~active
    run_id = (inactive != inactive.shift()).cumsum()
    run_length = inactive.groupby(run_id).transform("sum")
    bridgeable_silence = inactive & (run_length <= min_silence_frames)
    bridged_active = active | bridgeable_silence

    windows: list[ActivityWindow] = []
    window_id = (bridged_active != bridged_active.shift()).cumsum()
    for _, group in bridged_active.groupby(window_id):
        if not bool(group.iloc[0]):
            continue
        frames = group.index
        if len(frames) < min_window_frames:
            continue
        start_frame, end_frame = int(frames.min()), int(frames.max())
        windows.append(
            ActivityWindow(
                start_frame=start_frame,
                end_frame=end_frame,
                start_ms=float(timestamps.loc[start_frame]),
                end_ms=float(timestamps.loc[end_frame]),
            )
        )
    return windows


def _arbitrate_cross_hand(strikes: list[Strike], window_frames: int) -> None:
    """Physiquement, un seul bras frappe à la fois (jab/cross) : si gauche et droite
    sont confirmés dans la même fenêtre temporelle, ne garder que l'extension_ratio
    le plus élevé et rejeter l'autre (mute geometric_pass, le candidat reste visible)."""
    confirmed = sorted((s for s in strikes if s.geometric_pass), key=lambda s: s.frame_idx)

    i = 0
    while i < len(confirmed):
        cluster = [confirmed[i]]
        j = i + 1
        while j < len(confirmed) and confirmed[j].frame_idx - cluster[-1].frame_idx <= window_frames:
            cluster.append(confirmed[j])
            j += 1

        if len({s.hand for s in cluster}) > 1:
            winner = max(cluster, key=lambda s: s.extension_ratio or 0.0)
            for s in cluster:
                if s is not winner:
                    s.geometric_pass = False

        i = j


def detect_strikes(
    df: pd.DataFrame,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    min_interval_ms: float = DEFAULT_MIN_STRIKE_INTERVAL_MS,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    savgol_window: int = DEFAULT_SAVGOL_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    extension_prominence: float = DEFAULT_EXTENSION_PEAK_PROMINENCE,
    extension_threshold: float = DEFAULT_EXTENSION_RATIO_THRESHOLD,
    height_fraction: float = DEFAULT_HEAD_HEIGHT_TORSO_FRACTION,
    angle_threshold_deg: float = DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG,
    min_peak_speed: float = DEFAULT_MIN_PEAK_SPEED,
    geometric_window_frames: int = DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES,
    cross_hand_window_frames: int = DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES,
    min_peak_speed_by_hand: dict[str, float] | None = None,
) -> list[Strike]:
    """Détecte les coups pour les deux poignets (séries indépendantes) : un coup est
    un pic local d'extension du bras (jab/cross), confirmé par hauteur/direction/vitesse.
    Arbitrage croisé ensuite : gauche et droite ne peuvent pas être confirmés en même
    temps (un seul bras frappe à la fois). min_peak_speed_by_hand, si fourni, remplace
    min_peak_speed pour la main correspondante (calibration par main)."""
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(
        wide, max_gap_frames, min_visibility, savgol_window, savgol_polyorder
    )

    strikes: list[Strike] = []
    for hand, landmark_id in WRIST_LANDMARKS.items():
        wrist_speed = compute_wrist_speed(
            df, landmark_id, torso_velocity, max_gap_frames, min_visibility,
            savgol_window, savgol_polyorder,
        )
        geometry = compute_arm_geometry(
            wide, hand, min_visibility, max_gap_frames, savgol_window, savgol_polyorder
        )
        hand_min_peak_speed = (
            min_peak_speed_by_hand[hand] if min_peak_speed_by_hand else min_peak_speed
        )
        strikes.extend(
            detect_strikes_for_hand(
                geometry, wrist_speed, hand, min_interval_ms, extension_prominence,
                extension_threshold, height_fraction, angle_threshold_deg,
                hand_min_peak_speed, geometric_window_frames,
            )
        )

    strikes.sort(key=lambda s: s.timestamp_ms)
    _arbitrate_cross_hand(strikes, cross_hand_window_frames)
    return strikes


def detect_strikes_from_csv(csv_path: Path, **kwargs) -> list[Strike]:
    df = pd.read_csv(csv_path)
    return detect_strikes(df, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Détecte les coups (pics d'extension du bras, jab/cross) à partir du CSV d'une vidéo."
    )
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--csv-dir", type=Path, default=CSV_OUTPUT_DIR)
    parser.add_argument("--extension-threshold", type=float, default=DEFAULT_EXTENSION_RATIO_THRESHOLD)
    parser.add_argument("--extension-prominence", type=float, default=DEFAULT_EXTENSION_PEAK_PROMINENCE)
    parser.add_argument("--angle-threshold", type=float, default=DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG)
    parser.add_argument("--min-peak-speed", type=float, default=DEFAULT_MIN_PEAK_SPEED)
    parser.add_argument("--savgol-window", type=int, default=DEFAULT_SAVGOL_WINDOW)
    parser.add_argument("--savgol-polyorder", type=int, default=DEFAULT_SAVGOL_POLYORDER)
    parser.add_argument("--min-visibility", type=float, default=DEFAULT_MIN_VISIBILITY)
    parser.add_argument(
        "--cross-hand-window", type=int, default=DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES
    )
    args = parser.parse_args()

    csv_path = csv_path_for_video(args.video, args.csv_dir)
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path} (lancer landmark_extraction d'abord)")
        return

    strikes = detect_strikes_from_csv(
        csv_path,
        min_visibility=args.min_visibility,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        extension_threshold=args.extension_threshold,
        extension_prominence=args.extension_prominence,
        angle_threshold_deg=args.angle_threshold,
        min_peak_speed=args.min_peak_speed,
        cross_hand_window_frames=args.cross_hand_window,
    )
    left_count = sum(1 for s in strikes if s.hand == "left")
    right_count = sum(1 for s in strikes if s.hand == "right")
    left_confirmed = sum(1 for s in strikes if s.hand == "left" and s.geometric_pass)
    right_confirmed = sum(1 for s in strikes if s.hand == "right" and s.geometric_pass)
    print(
        f"{len(strikes)} coups détectés dans {csv_path.name} "
        f"({left_count} gauche, {right_count} droite) -- "
        f"filtre géométrique jab/cross confirmé : {left_confirmed} gauche, {right_confirmed} droite"
    )
    for strike in strikes:
        status = "confirmé" if strike.geometric_pass else "non confirmé"
        ext = f"{strike.extension_ratio:.3f}" if strike.extension_ratio is not None else "N/A"
        print(
            f"  [{strike.hand}] frame {strike.frame_idx} "
            f"({strike.timestamp_ms / 1000:.2f}s) speed={strike.speed:.3f} "
            f"ext={ext} ({status})"
        )


if __name__ == "__main__":
    main()
