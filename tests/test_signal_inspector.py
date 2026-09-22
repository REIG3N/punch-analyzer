import pytest

from punch_analyzer.signal_inspector import (
    find_stillness_cases,
    near_miss_speed_strikes,
    strikes_in_time_range,
)
from punch_analyzer.strike_detection import ActivityWindow, Strike


def _strike(hand: str, timestamp_ms: float, extension_ratio: float, speed: float) -> Strike:
    return Strike(
        hand=hand,
        frame_idx=int(timestamp_ms / 33.3),
        timestamp_ms=timestamp_ms,
        speed=speed,
        geometric_pass=extension_ratio >= 0.85 and speed >= 1.5,
        extension_ratio=extension_ratio,
    )


def test_strikes_in_time_range_filters_and_sorts_by_timestamp():
    strikes = [
        _strike("left", 5000.0, 0.9, 2.0),
        _strike("right", 500.0, 0.9, 2.0),
        _strike("left", 1500.0, 0.9, 2.0),
    ]

    result = strikes_in_time_range(strikes, 0.27, 1.81)

    assert [s.timestamp_ms for s in result] == [500.0, 1500.0]


def test_strikes_in_time_range_empty_when_nothing_matches():
    strikes = [_strike("left", 5000.0, 0.9, 2.0)]

    assert strikes_in_time_range(strikes, 0.27, 1.81) == []


def test_near_miss_speed_strikes_separates_near_perfect_from_below_speed():
    strikes = [
        _strike("left", 1000.0, 1.000, 1.427),  # près du seuil vitesse, extension parfaite
        _strike("right", 2000.0, 1.000, 1.6),  # extension parfaite, vitesse suffisante
        _strike("left", 3000.0, 0.5, 0.2),  # ni l'un ni l'autre
    ]

    near_perfect, below_speed = near_miss_speed_strikes(
        strikes, extension_floor=0.98, min_peak_speed=1.5
    )

    assert len(near_perfect) == 2
    assert [s.timestamp_ms for s in below_speed] == [1000.0]


def test_near_miss_speed_strikes_sorted_by_timestamp():
    strikes = [
        _strike("left", 3000.0, 1.0, 0.1),
        _strike("right", 1000.0, 1.0, 0.2),
    ]

    _near_perfect, below_speed = near_miss_speed_strikes(
        strikes, extension_floor=0.98, min_peak_speed=1.5
    )

    assert [s.timestamp_ms for s in below_speed] == [1000.0, 3000.0]


def test_find_stillness_cases_always_includes_first_window_of_session():
    windows = [ActivityWindow(start_frame=0, end_frame=30, start_ms=0.0, end_ms=1000.0)]
    strikes = [_strike("left", 500.0, 1.0, 1.2)]

    cases = find_stillness_cases(windows, strikes, min_gap_s=5.0)

    assert len(cases) == 1
    assert cases[0].gap_before_s is None
    assert cases[0].first_strike.timestamp_ms == 500.0


def test_find_stillness_cases_skips_normal_pause_between_combos():
    windows = [
        ActivityWindow(start_frame=0, end_frame=10, start_ms=0.0, end_ms=300.0),
        ActivityWindow(start_frame=60, end_frame=70, start_ms=2300.0, end_ms=2600.0),  # 2s de silence
    ]
    strikes = [_strike("left", 100.0, 1.0, 1.2), _strike("right", 2400.0, 1.0, 1.2)]

    cases = find_stillness_cases(windows, strikes, min_gap_s=5.0)

    # La 2e fenêtre a un silence de 2s avant elle, en dessous du seuil de 5s :
    # ce n'est pas une "longue immobilité", seulement une pause normale entre combos.
    assert len(cases) == 1
    assert cases[0].window_index == 1


def test_find_stillness_cases_includes_window_after_long_silence():
    windows = [
        ActivityWindow(start_frame=0, end_frame=10, start_ms=0.0, end_ms=300.0),
        ActivityWindow(start_frame=300, end_frame=310, start_ms=9000.0, end_ms=9300.0),  # 8.7s
    ]
    strikes = [_strike("left", 100.0, 1.0, 1.2), _strike("right", 10000.0, 1.0, 1.2)]

    cases = find_stillness_cases(windows, strikes, min_gap_s=5.0)

    assert len(cases) == 2
    assert cases[1].gap_before_s == pytest.approx(8.7)


def test_find_stillness_cases_skips_window_without_any_strike():
    windows = [ActivityWindow(start_frame=0, end_frame=30, start_ms=0.0, end_ms=1000.0)]

    cases = find_stillness_cases(windows, [], min_gap_s=5.0)

    assert cases == []


def test_stillness_case_subsequent_mean_speed():
    windows = [ActivityWindow(start_frame=0, end_frame=100, start_ms=0.0, end_ms=3000.0)]
    strikes = [
        _strike("left", 500.0, 1.0, 1.0),  # 1er coup, plus lent
        _strike("right", 1500.0, 1.0, 2.0),
        _strike("left", 2500.0, 1.0, 3.0),
    ]

    cases = find_stillness_cases(windows, strikes, min_gap_s=5.0)

    assert cases[0].first_strike.speed == 1.0
    assert cases[0].subsequent_mean_speed == pytest.approx(2.5)


def test_stillness_case_subsequent_mean_speed_none_when_single_strike():
    windows = [ActivityWindow(start_frame=0, end_frame=30, start_ms=0.0, end_ms=1000.0)]
    strikes = [_strike("left", 500.0, 1.0, 1.2)]

    cases = find_stillness_cases(windows, strikes, min_gap_s=5.0)

    assert cases[0].subsequent_mean_speed is None
