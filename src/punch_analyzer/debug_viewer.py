import argparse
from pathlib import Path

import cv2
import pandas as pd

from punch_analyzer.landmark_extraction import START_TIME_MS, draw_skeleton
from punch_analyzer.paths import CSV_OUTPUT_DIR, VIDEO_PATH, csv_path_for_video
from punch_analyzer.strike_detection import (
    LEFT_WRIST_ID,
    RIGHT_WRIST_ID,
    Strike,
    compute_wrist_speed,
    detect_strikes,
)

WINDOW_NAME = "Punch Analyzer - Debug"

STRIKE_FLASH_WINDOW_FRAMES = 3

PLAYBACK_SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0]
DEFAULT_SPEED_INDEX = PLAYBACK_SPEEDS.index(1.0)

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


def nearby_strike_hands(
    frame_idx: int, strikes: list[Strike], window_frames: int
) -> list[str]:
    return sorted(
        {s.hand for s in strikes if abs(s.frame_idx - frame_idx) <= window_frames}
    )


def format_overlay_lines(
    frame_idx: int,
    left_speed: float | None,
    right_speed: float | None,
    active_hands: list[str],
) -> list[str]:
    def fmt_speed(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "N/A"

    lines = [
        f"frame {frame_idx}",
        f"vitesse poignet gauche: {fmt_speed(left_speed)}",
        f"vitesse poignet droit: {fmt_speed(right_speed)}",
        "type de coup: N/A (Ticket 3)",
    ]
    if active_hands:
        lines.append(f"COUP DETECTE: {', '.join(active_hands)}")
    return lines


def draw_overlay(frame, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        y = 22 + i * 26
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _no_op(_value: int) -> None:
    pass


def run_debug_viewer(video_path: Path, output_dir: Path = CSV_OUTPUT_DIR) -> None:
    csv_path = csv_path_for_video(video_path, output_dir)
    if not csv_path.exists():
        print(f"CSV introuvable: {csv_path} (lancer landmark_extraction d'abord)")
        return

    df = pd.read_csv(csv_path)
    frame_landmarks = build_frame_landmarks(df)

    left_speed_lookup = build_speed_lookup(compute_wrist_speed(df, LEFT_WRIST_ID))
    right_speed_lookup = build_speed_lookup(compute_wrist_speed(df, RIGHT_WRIST_ID))
    strikes = detect_strikes(df)
    print(f"{len(strikes)} coups détectés (seuils par défaut de strike_detection)")
    print(CONTROLS_HELP)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"Erreur: vidéo non trouvée ou illisible ({video_path})")
        return

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    video_total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # frame_idx du CSV est relatif au début de l'extraction (START_TIME_MS
    # dans la vidéo), pas à la frame 0 absolue de la vidéo : on reproduit le
    # même seek que landmark_extraction.process_video() pour retrouver
    # l'offset réel, plutôt que de le recalculer via fps (arrondi imprécis).
    video.set(cv2.CAP_PROP_POS_MSEC, START_TIME_MS)
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

        active_hands = nearby_strike_hands(
            current_frame, strikes, STRIKE_FLASH_WINDOW_FRAMES
        )
        lines = format_overlay_lines(
            current_frame,
            left_speed_lookup.get(current_frame),
            right_speed_lookup.get(current_frame),
            active_hands,
        )
        draw_overlay(frame, lines)
        if active_hands:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 8)

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
    args = parser.parse_args()
    run_debug_viewer(args.video, args.csv_dir)


if __name__ == "__main__":
    main()
