import math

import pandas as pd

from punch_analyzer.debug_viewer import (
    build_frame_landmarks,
    build_speed_lookup,
    format_overlay_lines,
    nearby_strike_hands,
)
from punch_analyzer.strike_detection import Strike


def test_build_frame_landmarks_groups_by_frame_and_skips_nan():
    df = pd.DataFrame(
        [
            {"frame_idx": 0, "landmark_id": 15, "x": 0.1, "y": 0.2},
            {"frame_idx": 0, "landmark_id": 16, "x": 0.3, "y": 0.4},
            {"frame_idx": 1, "landmark_id": 15, "x": math.nan, "y": math.nan},
            {"frame_idx": 1, "landmark_id": 16, "x": 0.5, "y": 0.6},
        ]
    )

    frames = build_frame_landmarks(df)

    assert frames[0] == {15: (0.1, 0.2), 16: (0.3, 0.4)}
    assert frames[1] == {16: (0.5, 0.6)}
    assert 15 not in frames[1]


def test_build_speed_lookup_skips_nan_rows():
    df = pd.DataFrame(
        [
            {"frame_idx": 0, "speed": math.nan},
            {"frame_idx": 1, "speed": 1.23},
        ]
    )

    lookup = build_speed_lookup(df)

    assert lookup == {1: 1.23}


def test_nearby_strike_hands_within_window():
    strikes = [
        Strike(hand="left", frame_idx=10, timestamp_ms=100.0, speed=5.0),
        Strike(hand="right", frame_idx=50, timestamp_ms=500.0, speed=5.0),
    ]

    assert nearby_strike_hands(11, strikes, window_frames=3) == ["left"]
    assert nearby_strike_hands(30, strikes, window_frames=3) == []
    assert nearby_strike_hands(10, strikes, window_frames=0) == ["left"]


def test_format_overlay_lines_has_fixed_line_count_regardless_of_strikes():
    inactive = format_overlay_lines(0, None, None, [])
    active = format_overlay_lines(0, 0.5, 0.3, ["left", "right"])

    assert len(inactive) == len(active) == 4


def test_format_overlay_lines_marks_only_the_striking_hand():
    lines = format_overlay_lines(42, 0.5, None, ["left"])

    assert "frame 42" in lines[0]
    assert "gauche" in lines[1] and "0.500" in lines[1] and "COUP" in lines[1]
    assert "droit" in lines[2] and "N/A" in lines[2] and "COUP" not in lines[2]


def test_format_overlay_lines_no_marker_when_inactive():
    lines = format_overlay_lines(0, None, None, [])

    assert not any("COUP" in line for line in lines)
