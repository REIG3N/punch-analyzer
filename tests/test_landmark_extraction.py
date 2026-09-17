import math
from types import SimpleNamespace
import pytest
from punch_analyzer.landmark_extraction import TRACKED_INDICES, build_csv_rows

def make_landmark(x, y, z, visibility):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)

def make_detection_result_empty():
    return SimpleNamespace(pose_landmarks=[])

def make_detection_result_with_pose():
    landmarks = [make_landmark(x=i * 0.01, y=i * 0.02, z=i * 0.03, visibility=0.9) for i in range(33)]
    return SimpleNamespace(pose_landmarks=[landmarks])

def test_detection_empty():
    detection_result = make_detection_result_empty()
    result = build_csv_rows(detection_result, frame_idx=42, timestamp_ms=1400)
    for i in result:
        assert i["frame_idx"] == 42
        assert i["timestamp_ms"] == 1400
        assert i["landmark_id"] is not None
        assert math.isnan(i["x"])
        assert math.isnan(i["y"])
        assert math.isnan(i["z"])
        assert math.isnan(i["visibility"])
    assert len(result) == len(TRACKED_INDICES)
        
def test_detection_with_pose():
    detection_result = make_detection_result_with_pose()
    result = build_csv_rows(detection_result, frame_idx=42, timestamp_ms=1400)
    for row in result:
        n = row["landmark_id"]
        assert row["frame_idx"] == 42
        assert row["timestamp_ms"] == 1400
        assert row["landmark_id"] is not None
        assert row["x"] == n * 0.01
        assert row["y"] == n * 0.02
        assert row["z"] == n * 0.03
        assert row["visibility"] == 0.9
    assert len(result) == len(TRACKED_INDICES)