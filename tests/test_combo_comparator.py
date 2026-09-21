import json
from pathlib import Path

from punch_analyzer.combo_comparator import (
    ComboExpectation,
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
