import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from punch_analyzer.paths import CSV_OUTPUT_DIR, VIDEO_PATH, csv_path_for_video

LEFT_WRIST_ID = 15
RIGHT_WRIST_ID = 16
WRIST_LANDMARKS = {"left": LEFT_WRIST_ID, "right": RIGHT_WRIST_ID}

DEFAULT_MAX_GAP_FRAMES = 5
DEFAULT_MIN_STRIKE_INTERVAL_MS = 250.0
DEFAULT_MIN_PROMINENCE = 1.5


@dataclass
class Strike:
    hand: str
    frame_idx: int
    timestamp_ms: float
    speed: float


def _fill_short_gaps(series: pd.Series, max_gap_frames: int) -> pd.Series:
    is_na = series.isna()
    run_id = (is_na != is_na.shift()).cumsum()
    run_length = is_na.groupby(run_id).transform("sum")
    fillable = is_na & (run_length <= max_gap_frames)
    interpolated = series.interpolate(method="linear", limit_area="inside")
    return series.where(~fillable, interpolated)


def compute_wrist_speed(
    df: pd.DataFrame,
    landmark_id: int,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
) -> pd.DataFrame:
    """Vitesse (unités normalisées/s) d'un landmark poignet, dérivée de timestamp_ms."""
    wrist = (
        df[df["landmark_id"] == landmark_id]
        .sort_values("frame_idx")
        .reset_index(drop=True)
        .copy()
    )
    wrist["x"] = _fill_short_gaps(wrist["x"], max_gap_frames)
    wrist["y"] = _fill_short_gaps(wrist["y"], max_gap_frames)

    dx = wrist["x"].diff()
    dy = wrist["y"].diff()
    dt_seconds = wrist["timestamp_ms"].diff() / 1000.0

    distance = np.sqrt(dx**2 + dy**2)
    speed = distance / dt_seconds
    speed = speed.where(dt_seconds > 0)

    wrist["speed"] = speed
    return wrist


def detect_strikes_for_hand(
    wrist_speed: pd.DataFrame,
    hand: str,
    min_interval_ms: float = DEFAULT_MIN_STRIKE_INTERVAL_MS,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
) -> list[Strike]:
    valid = wrist_speed.dropna(subset=["speed"])
    if valid.empty:
        return []

    median_dt_ms = valid["timestamp_ms"].diff().median()
    if not median_dt_ms or median_dt_ms <= 0:
        return []
    min_distance_frames = max(1, round(min_interval_ms / median_dt_ms))

    peak_positions, _ = find_peaks(
        valid["speed"].to_numpy(),
        distance=min_distance_frames,
        prominence=min_prominence,
    )

    peaks = valid.iloc[peak_positions]
    return [
        Strike(
            hand=hand,
            frame_idx=int(row.frame_idx),
            timestamp_ms=float(row.timestamp_ms),
            speed=float(row.speed),
        )
        for row in peaks.itertuples()
    ]


def detect_strikes(
    df: pd.DataFrame,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    min_interval_ms: float = DEFAULT_MIN_STRIKE_INTERVAL_MS,
    min_prominence: float = DEFAULT_MIN_PROMINENCE,
) -> list[Strike]:
    """Détecte les coups pour les deux poignets, séries indépendantes."""
    strikes: list[Strike] = []
    for hand, landmark_id in WRIST_LANDMARKS.items():
        wrist_speed = compute_wrist_speed(df, landmark_id, max_gap_frames)
        strikes.extend(
            detect_strikes_for_hand(wrist_speed, hand, min_interval_ms, min_prominence)
        )
    return sorted(strikes, key=lambda s: s.timestamp_ms)


def detect_strikes_from_csv(csv_path: Path, **kwargs) -> list[Strike]:
    df = pd.read_csv(csv_path)
    return detect_strikes(df, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Détecte les coups (pics de vitesse du poignet) à partir du CSV d'une vidéo."
    )
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--csv-dir", type=Path, default=CSV_OUTPUT_DIR)
    args = parser.parse_args()

    csv_path = csv_path_for_video(args.video, args.csv_dir)
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path} (lancer landmark_extraction d'abord)")
        return

    strikes = detect_strikes_from_csv(csv_path)
    left_count = sum(1 for s in strikes if s.hand == "left")
    right_count = sum(1 for s in strikes if s.hand == "right")
    print(
        f"{len(strikes)} coups détectés dans {csv_path.name} "
        f"({left_count} gauche, {right_count} droite)"
    )
    for strike in strikes:
        print(
            f"  [{strike.hand}] frame {strike.frame_idx} "
            f"({strike.timestamp_ms / 1000:.2f}s) speed={strike.speed:.3f}"
        )


if __name__ == "__main__":
    main()
