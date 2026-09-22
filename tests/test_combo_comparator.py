import json
from pathlib import Path

from punch_analyzer.combo_comparator import (
    ComboExpectation,
    filter_buffer_strikes,
    filter_buffer_windows,
    format_window_report,
    load_combo_list,
    score_combos,
    strikes_in_window,
)
from punch_analyzer.strike_detection import ActivityWindow, Strike


def _strike(hand: str, frame_idx: int, geometric_pass: bool = True) -> Strike:
    return Strike(
        hand=hand,
        frame_idx=frame_idx,
        timestamp_ms=frame_idx * 33.3,
        speed=2.0,
        geometric_pass=geometric_pass,
        extension_ratio=0.95,
    )


def test_load_combo_list_reads_json(tmp_path: Path):
    combos_path = tmp_path / "combos.json"
    combos_path.write_text(
        json.dumps([{"name": "1 (jab)", "left": 1, "right": 0}, {"name": "1-2", "left": 1, "right": 1}])
    )

    combos = load_combo_list(combos_path)

    assert combos == [
        ComboExpectation(name="1 (jab)", left=1, right=0),
        ComboExpectation(name="1-2", left=1, right=1),
    ]


def test_strikes_in_window_filters_by_frame_range():
    strikes = [_strike("left", 5), _strike("right", 15), _strike("left", 25)]
    window = ActivityWindow(start_frame=10, end_frame=20, start_ms=0.0, end_ms=0.0)

    result = strikes_in_window(strikes, window)

    assert [s.frame_idx for s in result] == [15]


def test_score_combos_counts_confirmed_per_hand():
    windows = [
        ActivityWindow(start_frame=0, end_frame=10, start_ms=0.0, end_ms=333.0),
        ActivityWindow(start_frame=20, end_frame=30, start_ms=666.0, end_ms=1000.0),
    ]
    strikes = [
        _strike("left", 2),
        _strike("right", 5, geometric_pass=False),  # non confirmé, ne compte pas
        _strike("left", 22),
        _strike("right", 25),
    ]
    combos = [
        ComboExpectation(name="1 (jab)", left=1, right=0),
        ComboExpectation(name="1-2", left=1, right=1),
    ]

    scores = score_combos(windows, strikes, combos)

    assert scores[0].confirmed_left == 1
    assert scores[0].confirmed_right == 0
    assert scores[0].detected_right == 1  # détecté mais pas confirmé
    assert scores[0].left_match is True
    assert scores[0].right_match is True

    assert scores[1].confirmed_left == 1
    assert scores[1].confirmed_right == 1
    assert scores[1].left_match is True
    assert scores[1].right_match is True


def test_score_combos_flags_mismatch():
    windows = [ActivityWindow(start_frame=0, end_frame=10, start_ms=0.0, end_ms=333.0)]
    strikes = [_strike("left", 2), _strike("left", 4)]
    combos = [ComboExpectation(name="1 (jab)", left=1, right=0)]

    scores = score_combos(windows, strikes, combos)

    assert scores[0].confirmed_left == 2
    assert scores[0].left_match is False


def test_score_combos_zips_only_common_length():
    windows = [ActivityWindow(start_frame=0, end_frame=10, start_ms=0.0, end_ms=333.0)]
    combos = [
        ComboExpectation(name="1 (jab)", left=1, right=0),
        ComboExpectation(name="1-2", left=1, right=1),
    ]

    scores = score_combos(windows, [], combos)

    assert len(scores) == 1


def test_format_window_report_shows_duration_and_gap_since_previous():
    windows = [
        ActivityWindow(start_frame=0, end_frame=10, start_ms=1000.0, end_ms=1500.0),
        ActivityWindow(start_frame=50, end_frame=60, start_ms=4000.0, end_ms=4300.0),
    ]

    lines = format_window_report(windows)

    assert len(lines) == 2
    assert "1.00s" in lines[0] and "1.50s" in lines[0] and "0.50s" in lines[0]
    assert "N/A" in lines[0]  # pas de fenêtre précédente
    assert "4.00s" in lines[1] and "4.30s" in lines[1]
    assert "2.50s" in lines[1]  # silence = 4.00s - 1.50s


def test_score_combos_keeps_individual_strikes_sorted_for_peak_speed_reporting():
    windows = [ActivityWindow(start_frame=0, end_frame=30, start_ms=0.0, end_ms=1000.0)]
    strikes = [
        _strike("right", 20),
        _strike("left", 5),
    ]
    combos = [ComboExpectation(name="1-2", left=1, right=1)]

    scores = score_combos(windows, strikes, combos)

    assert [s.frame_idx for s in scores[0].strikes] == [5, 20]
    assert [s.hand for s in scores[0].strikes] == ["left", "right"]


def test_filter_buffer_windows_drops_windows_entirely_inside_buffer():
    windows = [
        ActivityWindow(start_frame=0, end_frame=5, start_ms=0.0, end_ms=1000.0),  # dans la marge
        ActivityWindow(start_frame=10, end_frame=20, start_ms=2000.0, end_ms=3000.0),  # après
    ]

    result = filter_buffer_windows(windows, buffer_ms=2500.0)

    assert len(result) == 1
    assert result[0].start_ms == 2000.0


def test_filter_buffer_windows_keeps_window_straddling_the_boundary():
    windows = [ActivityWindow(start_frame=0, end_frame=20, start_ms=2000.0, end_ms=3000.0)]

    result = filter_buffer_windows(windows, buffer_ms=2500.0)

    assert len(result) == 1


def test_filter_buffer_strikes_drops_strikes_before_buffer():
    strikes = [
        Strike(hand="left", frame_idx=30, timestamp_ms=1000.0, speed=2.0, geometric_pass=True),
        Strike(hand="right", frame_idx=90, timestamp_ms=3000.0, speed=2.0, geometric_pass=True),
    ]

    result = filter_buffer_strikes(strikes, buffer_ms=2500.0)

    assert [s.timestamp_ms for s in result] == [3000.0]


def test_format_window_report_shifts_display_by_buffer_ms():
    windows = [ActivityWindow(start_frame=0, end_frame=10, start_ms=2800.0, end_ms=3200.0)]

    lines = format_window_report(windows, buffer_ms=2500.0)

    assert "0.30s" in lines[0] and "0.70s" in lines[0]
