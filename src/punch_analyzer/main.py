import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ASSET_PATH = PROJECT_ROOT / "data" / "pose_landmarker_full.task"
VIDEO_PATH = PROJECT_ROOT / "video" / "videoplayback.mp4"
CSV_OUTPUT_PATH = PROJECT_ROOT / "data" / "csv" / "landmarks_raw.csv"

START_TIME_MS = (
    120000  # commence à 120 s (ajustez selon votre timing d'intro + corde à sauter)
)
SMOOTHING_ALPHA = 0.9  # 0 = très lisse mais lent à réagir, 1 = pas de lissage

TRACKED_INDICES = [
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]
CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
]


def load_landmarker() -> vision.PoseLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(MODEL_ASSET_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_skeleton(frame, smoothed_landmarks, width: int, height: int) -> None:
    for idx in TRACKED_INDICES:
        if idx in smoothed_landmarks:
            new_x, new_y = smoothed_landmarks[idx]
            cx = int(new_x * width)
            cy = int(new_y * height)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    for start_idx, end_idx in CONNECTIONS:
        if start_idx in smoothed_landmarks and end_idx in smoothed_landmarks:
            x1, y1 = smoothed_landmarks[start_idx]
            x2, y2 = smoothed_landmarks[end_idx]
            pt1 = (int(x1 * width), int(y1 * height))
            pt2 = (int(x2 * width), int(y2 * height))
            cv2.line(frame, pt1, pt2, (0, 200, 255), 2)


def process_video(landmarker: vision.PoseLandmarker, video_path: Path) -> None:
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print("Erreur: vidéo non trouvée ou illisible")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_MSEC, START_TIME_MS)
    frame_idx = 0
    csv_rows = []

    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_idx * 1000 / fps)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            for idx in TRACKED_INDICES:
                lm = landmarks[idx]
                csv_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "landmark_id": idx,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
                )

        frame_idx += 1

    video.release()

    if csv_rows:
        fieldnames = [
            "frame_idx",
            "timestamp_ms",
            "landmark_id",
            "x",
            "y",
            "z",
            "visibility",
        ]
        with open(CSV_OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV written to {CSV_OUTPUT_PATH} ({len(csv_rows)} rows)")
    else:
        print("No landmarks detected, CSV not written")


def main() -> None:
    with load_landmarker() as landmarker:
        process_video(landmarker, VIDEO_PATH)


if __name__ == "__main__":
    main()
