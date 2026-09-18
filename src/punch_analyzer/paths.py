from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ASSET_PATH = PROJECT_ROOT / "data" / "pose_landmarker_full.task"
VIDEO_PATH = PROJECT_ROOT / "video" / "videoplayback.mp4"
CSV_OUTPUT_DIR = PROJECT_ROOT / "data" / "csv"


def csv_path_for_video(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{video_path.stem}.csv"
