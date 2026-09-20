import math

import pandas as pd

from punch_analyzer.debug_viewer import (
    build_extension_lookup,
    build_frame_landmarks,
    build_speed_lookup,
    format_overlay_lines,
    nearby_strikes,
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


def test_build_extension_lookup_skips_nan_rows():
    df = pd.DataFrame(
        [
            {"frame_idx": 0, "extension_ratio": math.nan},
            {"frame_idx": 1, "extension_ratio": 0.91},
        ]
    )

    lookup = build_extension_lookup(df)

    assert lookup == {1: 0.91}


def test_nearby_strikes_within_window():
    strikes = [
        Strike(hand="left", frame_idx=10, timestamp_ms=100.0, speed=5.0, geometric_pass=True),
        Strike(hand="right", frame_idx=50, timestamp_ms=500.0, speed=5.0, geometric_pass=False),
    ]

    assert [s.hand for s in nearby_strikes(11, strikes, window_frames=3)] == ["left"]
    assert nearby_strikes(30, strikes, window_frames=3) == []
    assert [s.hand for s in nearby_strikes(10, strikes, window_frames=0)] == ["left"]


def test_format_overlay_lines_has_fixed_line_count_regardless_of_strikes():
    inactive = format_overlay_lines(0, None, None, None, None, [])
    active = format_overlay_lines(
        0, 0.5, 0.3, 0.9, 0.6,
        [Strike(hand="left", frame_idx=0, timestamp_ms=0.0, speed=5.0, geometric_pass=True)],
    )

    assert len(inactive) == len(active) == 4


def test_format_overlay_lines_marks_confirmed_strike():
    strikes = [
        Strike(hand="left", frame_idx=42, timestamp_ms=0.0, speed=5.0, geometric_pass=True)
    ]
    lines = format_overlay_lines(42, 0.5, None, 0.95, None, strikes)

    assert "frame 42" in lines[0]
    assert "gauche" in lines[1] and "0.500" in lines[1] and "COUP" in lines[1]
    assert "droit" in lines[2] and "N/A" in lines[2] and "COUP" not in lines[2]


def test_format_overlay_lines_marks_unconfirmed_strike_differently():
    strikes = [
        Strike(hand="left", frame_idx=42, timestamp_ms=0.0, speed=5.0, geometric_pass=False)
    ]
    lines = format_overlay_lines(42, 0.5, None, 0.6, None, strikes)

    assert "coup?" in lines[1]
    assert "-- COUP" not in lines[1]


def test_format_overlay_lines_no_marker_when_inactive():
    lines = format_overlay_lines(0, None, None, None, None, [])

    assert not any("COUP" in line or "coup?" in line for line in lines)
