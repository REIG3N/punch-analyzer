import argparse
from pathlib import Path

import cv2
import pandas as pd

from punch_analyzer.landmark_extraction import draw_skeleton
from punch_analyzer.paths import CSV_OUTPUT_DIR, VIDEO_PATH, csv_path_for_video
from punch_analyzer.strike_detection import (
    DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES,
    DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG,
    DEFAULT_EXTENSION_PEAK_PROMINENCE,
    DEFAULT_EXTENSION_RATIO_THRESHOLD,
    DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES,
    DEFAULT_HEAD_HEIGHT_TORSO_FRACTION,
    DEFAULT_MIN_PEAK_SPEED,
    DEFAULT_MIN_VISIBILITY,
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW,
    LEFT_WRIST_ID,
    RIGHT_WRIST_ID,
    Strike,
    compute_arm_geometry,
    compute_torso_center_velocity,
    compute_wrist_speed,
    detect_strikes,
    pivot_landmarks,
)

WINDOW_NAME = "Punch Analyzer - Debug"

STRIKE_FLASH_WINDOW_FRAMES = 3

PLAYBACK_SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0]
DEFAULT_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)

LEFT_STRIKE_COLOR = (0, 0, 255)  # rouge (BGR)
RIGHT_STRIKE_COLOR = (255, 0, 0)  # bleu (BGR)
STRIKE_BORDER_THICKNESS = 10
UNCONFIRMED_BORDER_THICKNESS = 4

CONTROLS_HELP = (
    "Contrôles: [espace] pause/lecture | a/d frame précédente/suivante "
    "(en pause) | [ ] vitesse de lecture -/+ | trackbar = position | "
    "q/esc quitter"
)


def build_frame_landmarks(
    df: pd.DataFrame,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Reconstruit {frame_idx: {landmark_id: (x, y)}} à partir du CSV long, format attendu par draw_skeleton."""
    frames: dict[int, dict[int, tuple[float, float]]] = {}
    valid = df.dropna(subset=["x", "y"])
    for frame_idx, group in valid.groupby("frame_idx"):
        frames[int(frame_idx)] = {
            int(row.landmark_id): (float(row.x), float(row.y))
            for row in group.itertuples()
        }
    return frames


def build_speed_lookup(wrist_speed: pd.DataFrame) -> dict[int, float]:
    valid = wrist_speed.dropna(subset=["speed"])
    return {int(row.frame_idx): float(row.speed) for row in valid.itertuples()}


def build_extension_lookup(geometry: pd.DataFrame) -> dict[int, float]:
    valid = geometry.dropna(subset=["extension_ratio"])
    return {int(row.frame_idx): float(row.extension_ratio) for row in valid.itertuples()}


def nearby_strikes(
    frame_idx: int, strikes: list[Strike], window_frames: int
) -> list[Strike]:
    return [s for s in strikes if abs(s.frame_idx - frame_idx) <= window_frames]


def format_overlay_lines(
    frame_idx: int,
    left_speed: float | None,
    right_speed: float | None,
    left_extension: float | None,
    right_extension: float | None,
    active_strikes: list[Strike],
) -> list[str]:
    """4 lignes fixes : le marqueur de coup s'ajoute à la ligne de la main
    concernée plutôt que d'apparaître/disparaître comme ligne séparée, pour
    que rien ne se décale à l'écran quand un coup est détecté."""

    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "N/A"

    def marker(hand: str) -> str:
        for strike in active_strikes:
            if strike.hand == hand:
                return " -- COUP" if strike.geometric_pass else " -- coup?"
        return ""

    return [
        f"frame {frame_idx}",
        f"gauche: v={fmt(left_speed)} ext={fmt(left_extension)}{marker('left')}",
        f"droit: v={fmt(right_speed)} ext={fmt(right_extension)}{marker('right')}",
        "type de coup: N/A (Ticket 3)",
    ]


def draw_overlay(frame, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 20
    padding = 8
    # Marge à gauche pour ne pas passer sous la bordure de coup détecté
    # (draw_strike_borders), dessinée après l'overlay pour rester saturée.
    text_left = STRIKE_BORDER_THICKNESS + padding

    text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines
    )
    box_width = text_left + text_width + padding
    box_height = line_height * len(lines) + padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = padding + line_height * (i + 1) - 6
        cv2.putText(
            frame,
            line,
            (text_left, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_strike_borders(frame, active_strikes: list[Strike], width: int, height: int) -> None:
    for hand, color in (("left", LEFT_STRIKE_COLOR), ("right", RIGHT_STRIKE_COLOR)):
        strikes = [s for s in active_strikes if s.hand == hand]
        if not strikes:
            continue
        confirmed = any(s.geometric_pass for s in strikes)
        border_thickness = STRIKE_BORDER_THICKNESS if confirmed else UNCONFIRMED_BORDER_THICKNESS
        x0 = 0 if hand == "left" else width - 1 - border_thickness
        x1 = border_thickness if hand == "left" else width - 1
        cv2.rectangle(frame, (x0, 0), (x1, height - 1), color, -1)


def _no_op(_value: int) -> None:
    pass


def run_debug_viewer(
    video_path: Path,
    output_dir: Path = CSV_OUTPUT_DIR,
    start_ms: int | None = None,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
    savgol_window: int = DEFAULT_SAVGOL_WINDOW,
    savgol_polyorder: int = DEFAULT_SAVGOL_POLYORDER,
    geometric_window_frames: int = DEFAULT_GEOMETRIC_CONFIRM_WINDOW_FRAMES,
    extension_threshold: float = DEFAULT_EXTENSION_RATIO_THRESHOLD,
    extension_prominence: float = DEFAULT_EXTENSION_PEAK_PROMINENCE,
    height_fraction: float = DEFAULT_HEAD_HEIGHT_TORSO_FRACTION,
    angle_threshold_deg: float = DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG,
    min_peak_speed: float = DEFAULT_MIN_PEAK_SPEED,
    cross_hand_window_frames: int = DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES,
) -> None:
    csv_path = csv_path_for_video(video_path, output_dir)
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path} (lancer landmark_extraction d'abord)")
        return

    df = pd.read_csv(csv_path)
    frame_landmarks = build_frame_landmarks(df)

    wide = pivot_landmarks(df)
    torso_velocity = compute_torso_center_velocity(wide, min_visibility=min_visibility,
                                                     savgol_window=savgol_window,
                                                     savgol_polyorder=savgol_polyorder)
    left_speed_lookup = build_speed_lookup(
        compute_wrist_speed(df, LEFT_WRIST_ID, torso_velocity, min_visibility=min_visibility,
                             savgol_window=savgol_window, savgol_polyorder=savgol_polyorder)
    )
    right_speed_lookup = build_speed_lookup(
        compute_wrist_speed(df, RIGHT_WRIST_ID, torso_velocity, min_visibility=min_visibility,
                             savgol_window=savgol_window, savgol_polyorder=savgol_polyorder)
    )
    left_extension_lookup = build_extension_lookup(
        compute_arm_geometry(wide, "left", min_visibility,
                              savgol_window=savgol_window, savgol_polyorder=savgol_polyorder)
    )
    right_extension_lookup = build_extension_lookup(
        compute_arm_geometry(wide, "right", min_visibility,
                              savgol_window=savgol_window, savgol_polyorder=savgol_polyorder)
    )

    strikes = detect_strikes(
        df,
        min_visibility=min_visibility,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        geometric_window_frames=geometric_window_frames,
        extension_threshold=extension_threshold,
        extension_prominence=extension_prominence,
        height_fraction=height_fraction,
        angle_threshold_deg=angle_threshold_deg,
        min_peak_speed=min_peak_speed,
        cross_hand_window_frames=cross_hand_window_frames,
    )
    confirmed_count = sum(1 for s in strikes if s.geometric_pass)
    print(
        f"{len(strikes)} coups détectés (pics d'extension du bras), "
        f"{confirmed_count} confirmés par le filtre géométrique jab/cross"
    )
    print(
        f"seuils actifs : extension>{extension_threshold} (prominence={extension_prominence}) "
        f"angle<{angle_threshold_deg}° min_peak_speed={min_peak_speed} "
        f"savgol_window={savgol_window} savgol_polyorder={savgol_polyorder} "
        f"min_visibility={min_visibility}"
    )
    print(CONTROLS_HELP)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"Erreur: vidéo non trouvée ou illisible ({video_path})")
        return

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    video_total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # frame_idx du CSV est relatif au début de l'extraction (start_ms dans
    # la vidéo), pas à la frame 0 absolue de la vidéo : on reproduit le
    # même seek que landmark_extraction.process_video() pour retrouver
    # l'offset réel, plutôt que de le recalculer via fps (arrondi imprécis).
    video.set(cv2.CAP_PROP_POS_MSEC, start_ms or 0)
    frame_offset = int(video.get(cv2.CAP_PROP_POS_FRAMES))

    csv_frame_count = int(df["frame_idx"].max()) + 1 if not df.empty else 0
    total_frames = max(0, min(csv_frame_count, video_total_frames - frame_offset))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.createTrackbar("Frame", WINDOW_NAME, 0, max(total_frames - 1, 0), _no_op)

    speed_index = DEFAULT_SPEED_INDEX
    playing = True
    current_frame = 0

    while True:
        current_frame = cv2.getTrackbarPos("Frame", WINDOW_NAME)

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_offset + current_frame)
        ret, frame = video.read()
        if not ret:
            break

        draw_skeleton(frame, frame_landmarks.get(current_frame, {}), width, height)

        active_strikes = nearby_strikes(current_frame, strikes, STRIKE_FLASH_WINDOW_FRAMES)
        lines = format_overlay_lines(
            current_frame,
            left_speed_lookup.get(current_frame),
            right_speed_lookup.get(current_frame),
            left_extension_lookup.get(current_frame),
            right_extension_lookup.get(current_frame),
            active_strikes,
        )
        draw_overlay(frame, lines)
        draw_strike_borders(frame, active_strikes, width, height)

        cv2.imshow(WINDOW_NAME, frame)

        delay_ms = (
            max(1, int(1000 / (fps * PLAYBACK_SPEEDS[speed_index]))) if playing else 30
        )
        key = cv2.waitKey(delay_ms) & 0xFF

        next_frame = current_frame
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            playing = False
            next_frame = min(current_frame + 1, total_frames - 1)
        elif key == ord("a"):
            playing = False
            next_frame = max(current_frame - 1, 0)
        elif key == ord("]"):
            speed_index = min(speed_index + 1, len(PLAYBACK_SPEEDS) - 1)
        elif key == ord("["):
            speed_index = max(speed_index - 1, 0)
        elif playing:
            next_frame = min(current_frame + 1, total_frames - 1)
            if next_frame == current_frame:
                playing = False

        cv2.setTrackbarPos("Frame", WINDOW_NAME, next_frame)

    video.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outil de debug: squelette + vitesse poignet + coups détectés, superposés sur la vidéo source."
    )
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--csv-dir", type=Path, default=CSV_OUTPUT_DIR)
    parser.add_argument("--start-ms", type=int, default=None)
    parser.add_argument("--min-visibility", type=float, default=DEFAULT_MIN_VISIBILITY)
    parser.add_argument("--savgol-window", type=int, default=DEFAULT_SAVGOL_WINDOW)
    parser.add_argument("--savgol-polyorder", type=int, default=DEFAULT_SAVGOL_POLYORDER)
    parser.add_argument("--extension-threshold", type=float, default=DEFAULT_EXTENSION_RATIO_THRESHOLD)
    parser.add_argument("--extension-prominence", type=float, default=DEFAULT_EXTENSION_PEAK_PROMINENCE)
    parser.add_argument("--angle-threshold", type=float, default=DEFAULT_DIRECTION_ANGLE_THRESHOLD_DEG)
    parser.add_argument("--min-peak-speed", type=float, default=DEFAULT_MIN_PEAK_SPEED)
    parser.add_argument(
        "--cross-hand-window", type=int, default=DEFAULT_CROSS_HAND_ARBITRATION_WINDOW_FRAMES
    )
    args = parser.parse_args()
    run_debug_viewer(
        args.video,
        args.csv_dir,
        args.start_ms,
        min_visibility=args.min_visibility,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
        extension_threshold=args.extension_threshold,
        extension_prominence=args.extension_prominence,
        angle_threshold_deg=args.angle_threshold,
        min_peak_speed=args.min_peak_speed,
        cross_hand_window_frames=args.cross_hand_window,
    )


if __name__ == "__main__":
    main()
