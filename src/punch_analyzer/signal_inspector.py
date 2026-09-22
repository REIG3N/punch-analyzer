import argparse
from pathlib import Path

import pandas as pd

from punch_analyzer.paths import CSV_OUTPUT_DIR, VIDEO_PATH, csv_path_for_video
from punch_analyzer.strike_detection import (
    DEFAULT_ACTIVITY_THRESHOLD,
    DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES,
    DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG,
    DEFAULT_EXTENSION_PEAK_PROMINENCE,
    DEFAULT_EXTENSION_RATIO_THRESHOLD,
    DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES,
    DEFAULT_HEAD_HEIGHT_TORSO_FRACTION,
    DEFAULT_MAX_GAP_FRAMES,
    DEFAULT_MIN_PEAK_SPEED,
    DEFAULT_MIN_VISIBILITY,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW,
    WRIST_LANDMARKS,
    Strike,
    compute_arm_geometry,
    compute_torso_center_velocity,
    compute_wrist_speed,
    detect_strikes,
    pivot_landmarks,
)

# Outil de diagnostic en lecture seule : n'appelle jamais detect_strikes() ou les
# fonctions de bas niveau avec des seuils différents de ceux fournis en CLI --
# les valeurs par défaut ici sont celles actuellement actives dans le pipeline,
# pas des propositions.


def _load_pipeline(
    csv_path: Path,
    max_gap_frames: int,
    min_visibility: float,
    savgol_window: int,
    savgol_polyorder: int,
    **detect_kwargs,
):
    df = pd.read_csv(csv_path)
    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(
        wide, max_gap_frames, min_visibility, savgol_window, savgol_polyorder
    )
    wrist_speed = {
        hand: compute_wrist_speed(
            df, landmark_id, torso_velocity, max_gap_frames, min_visibility,
            savgol_window, savgol_polyorder,
        )
        for hand, landmark_id in WRIST_LANDMARKS.items()
    }
    geometry = {
        hand: compute_arm_geometry(
            wide, hand, min_visibility, max_gap_frames, savgol_window, savgol_polyorder
        )
        for hand in WRIST_LANDMARKS
    }
    strikes = detect_strikes(
        df, max_gap_frames=max_gap_frames, min_visibility=min_visibility,
        savgol_window=savgol_window, savgol_polyorder=savgol_polyorder, **detect_kwargs,
    )
    return wide, wrist_speed, geometry, strikes


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--csv-dir", type=Path, default=CSV_OUTPUT_DIR)
    parser.add_argument(
        "--buffer-ms", type=float, default=0,
        help="Même valeur que --buffer-ms passé à landmark_extraction pour ce CSV -- "
        "--start/--end restent relatifs à start_ms, décalés en interne vers le CSV brut.",
    )
    parser.add_argument("--max-gap-frames", type=int, default=DEFAULT_MAX_GAP_FRAMES)
    parser.add_argument("--min-visibility", type=float, default=DEFAULT_MIN_VISIBILITY)
    parser.add_argument("--savgol-window", type=int, default=DEFAULT_SAVGOL_WINDOW)
    parser.add_argument("--savgol-polyorder", type=int, default=DEFAULT_SAVGOL_POLYORDER)
    parser.add_argument("--extension-threshold", type=float, default=DEFAULT_EXTENSION_RATIO_THRESHOLD)
    parser.add_argument("--extension-prominence", type=float, default=DEFAULT_EXTENSION_PEAK_PROMINENCE)
    parser.add_argument("--height-fraction", type=float, default=DEFAULT_HEAD_HEIGHT_TORSO_FRACTION)
    parser.add_argument("--angle-threshold", type=float, default=DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG)
    parser.add_argument("--min-peak-speed", type=float, default=DEFAULT_MIN_PEAK_SPEED)
    parser.add_argument(
        "--geometric-window-frames", type=int, default=DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES
    )
    parser.add_argument(
        "--cross-hand-window", type=int, default=DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES
    )


def _detect_kwargs(args: argparse.Namespace) -> dict:
    return {
        "extension_threshold": args.extension_threshold,
        "extension_prominence": args.extension_prominence,
        "height_fraction": args.height_fraction,
        "angle_threshold_deg": args.angle_threshold,
        "min_peak_speed": args.min_peak_speed,
        "geometric_window_frames": args.geometric_window_frames,
        "cross_hand_window_frames": args.cross_hand_window,
    }


def strikes_in_time_range(strikes: list[Strike], start_s: float, end_s: float) -> list[Strike]:
    return sorted(
        (s for s in strikes if start_s <= s.timestamp_ms / 1000 <= end_s),
        key=lambda s: s.timestamp_ms,
    )


def near_miss_speed_strikes(
    strikes: list[Strike], extension_floor: float, min_peak_speed: float
) -> tuple[list[Strike], list[Strike]]:
    """Retourne (candidats à extension quasi parfaite, ceux parmi eux sous min_peak_speed)."""
    near_perfect = [
        s for s in strikes if s.extension_ratio is not None and s.extension_ratio >= extension_floor
    ]
    below_speed = sorted(
        (s for s in near_perfect if s.speed < min_peak_speed), key=lambda s: s.timestamp_ms
    )
    return near_perfect, below_speed


def cmd_peaks_in_range(args: argparse.Namespace) -> None:
    """Point 1 : tous les candidats d'extension (confirmés ou non) dans une plage de
    temps, indépendamment du fenêtrage d'activité, + le signal d'activité brut du
    segmenteur sur cette même plage pour voir s'il franchit activity_threshold."""
    csv_path = csv_path_for_video(args.video, args.csv_dir)
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path} (lancer landmark_extraction d'abord)")
        return

    wide, wrist_speed, _geometry, strikes = _load_pipeline(
        csv_path, args.max_gap_frames, args.min_visibility, args.savgol_window,
        args.savgol_polyorder, **_detect_kwargs(args),
    )

    buffer_s = args.buffer_ms / 1000
    raw_start, raw_end = args.start + buffer_s, args.end + buffer_s

    print(f"Candidats d'extension entre {args.start}s et {args.end}s (tous, confirmés ou non) :")
    in_range = strikes_in_time_range(strikes, raw_start, raw_end)
    if not in_range:
        print("  (aucun -- aucun pic d'extension_ratio n'a même été détecté par find_peaks ici)")
    for s in in_range:
        status = "confirmé" if s.geometric_pass else "non confirmé"
        print(
            f"  [{s.hand}] {s.timestamp_ms / 1000 - buffer_s:.3f}s ext={s.extension_ratio:.4f} "
            f"speed={s.speed:.3f} ({status})"
        )

    left_speed = wrist_speed["left"].set_index("frame_idx")["speed"]
    right_speed = wrist_speed["right"].set_index("frame_idx")["speed"]
    combined = pd.concat([left_speed, right_speed], axis=1).max(axis=1, skipna=True).fillna(0.0)
    timestamps = wide.set_index("frame_idx")["timestamp_ms"]

    print(
        f"\nSignal d'activité du segmenteur (max vitesse gauche/droite par frame) "
        f"entre {args.start}s et {args.end}s (seuil : {args.activity_threshold}) :"
    )
    for frame_idx, t_ms in timestamps.items():
        t = t_ms / 1000
        if raw_start <= t <= raw_end:
            value = combined.get(frame_idx, float("nan"))
            flag = "AU-DESSUS" if value > args.activity_threshold else "en dessous"
            print(f"  frame {frame_idx} {t - buffer_s:.3f}s activité={value:.3f} ({flag})")


def cmd_extension_curve(args: argparse.Namespace) -> None:
    """Point 2 : extension_ratio brut par frame pour une main, sur une plage de
    temps -- pas juste la valeur au pic, toute la courbe."""
    csv_path = csv_path_for_video(args.video, args.csv_dir)
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path} (lancer landmark_extraction d'abord)")
        return

    _wide, _wrist_speed, geometry, _strikes = _load_pipeline(
        csv_path, args.max_gap_frames, args.min_visibility, args.savgol_window,
        args.savgol_polyorder, **_detect_kwargs(args),
    )

    buffer_s = args.buffer_ms / 1000
    raw_start, raw_end = args.start + buffer_s, args.end + buffer_s

    print(f"extension_ratio brut, bras {args.hand}, {args.start}s -> {args.end}s :")
    for row in geometry[args.hand].itertuples():
        t = row.timestamp_ms / 1000
        if raw_start <= t <= raw_end:
            ext = row.extension_ratio
            ext_str = f"{ext:.4f}" if pd.notna(ext) else "NaN"
            print(f"  frame {row.frame_idx} {t - buffer_s:.3f}s ext={ext_str}")


def cmd_near_miss_speed(args: argparse.Namespace) -> None:
    """Point 3 : coups à extension quasi parfaite mais vitesse sous le seuil --
    fréquent = piste savgol_window trop lissant ; rare = piste min_peak_speed."""
    csv_path = csv_path_for_video(args.video, args.csv_dir)
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path} (lancer landmark_extraction d'abord)")
        return

    _wide, _wrist_speed, _geometry, strikes = _load_pipeline(
        csv_path, args.max_gap_frames, args.min_visibility, args.savgol_window,
        args.savgol_polyorder, **_detect_kwargs(args),
    )

    buffer_s = args.buffer_ms / 1000
    strikes = [s for s in strikes if s.timestamp_ms >= args.buffer_ms]
    near_perfect, below_speed = near_miss_speed_strikes(
        strikes, args.extension_floor, args.min_peak_speed
    )

    print(
        f"{len(below_speed)} coup(s) avec extension_ratio >= {args.extension_floor} mais "
        f"speed < min_peak_speed ({args.min_peak_speed}), sur {len(near_perfect)} candidats "
        f"à extension quasi parfaite au total (marge exclue) :"
    )
    for s in below_speed:
        print(
            f"  [{s.hand}] {s.timestamp_ms / 1000 - buffer_s:.2f}s ext={s.extension_ratio:.4f} "
            f"speed={s.speed:.3f} (manque {args.min_peak_speed - s.speed:.3f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspection brute du signal (extension_ratio, vitesse) pour diagnostiquer "
        "un coup raté avant de retoucher un seuil."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    peaks_parser = subparsers.add_parser(
        "peaks-in-range", help="Liste les candidats d'extension dans une plage de temps."
    )
    peaks_parser.add_argument("--start", type=float, required=True, help="Début (secondes)")
    peaks_parser.add_argument("--end", type=float, required=True, help="Fin (secondes)")
    peaks_parser.add_argument("--activity-threshold", type=float, default=DEFAULT_ACTIVITY_THRESHOLD)
    _add_common_args(peaks_parser)
    peaks_parser.set_defaults(func=cmd_peaks_in_range)

    curve_parser = subparsers.add_parser(
        "extension-curve", help="Courbe brute d'extension_ratio pour une main, sur une plage de temps."
    )
    curve_parser.add_argument("--hand", choices=["left", "right"], required=True)
    curve_parser.add_argument("--start", type=float, required=True, help="Début (secondes)")
    curve_parser.add_argument("--end", type=float, required=True, help="Fin (secondes)")
    _add_common_args(curve_parser)
    curve_parser.set_defaults(func=cmd_extension_curve)

    near_miss_parser = subparsers.add_parser(
        "near-miss-speed", help="Coups à extension quasi parfaite mais vitesse sous le seuil."
    )
    near_miss_parser.add_argument("--extension-floor", type=float, default=0.98)
    _add_common_args(near_miss_parser)
    near_miss_parser.set_defaults(func=cmd_near_miss_speed)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
