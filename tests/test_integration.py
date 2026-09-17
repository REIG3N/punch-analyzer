import csv
from pathlib import Path

from punch_analyzer.landmark_extraction import (
    TRACKED_INDICES,
    VIDEO_PATH,
    load_landmarker,
    process_video,
)

TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"


def test_process_video_end_to_end():
    with load_landmarker() as landmarker:
        process_video(landmarker, VIDEO_PATH, output_dir=TEST_OUTPUT_DIR)

    output_path = TEST_OUTPUT_DIR / f"{VIDEO_PATH.stem}.csv"
    assert output_path.exists()

    with open(output_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) > 0
    assert len(rows) % len(TRACKED_INDICES) == 0